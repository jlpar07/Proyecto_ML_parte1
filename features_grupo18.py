"""Features del Grupo 18 - Proyecto ML parte 1 (envio final).

Este modulo se importa desde el notebook para que el modelo entrenado se pueda
guardar y volver a cargar con joblib/pickle: una clase definida dentro del
notebook no se puede deserializar en otra maquina.

TODO lo que hay aqui es procesamiento de texto clasico (separacion en clausulas,
conteos de vocabulario, TF-IDF de palabras y de caracteres, n-gramas) y modelos
de scikit-learn. No se usa aprendizaje profundo, ni arquitecturas transformer,
ni embeddings preentrenados, conforme a los requisitos de la Parte 1.

========================= El problema y la idea =========================

Cada resena mezcla varias cláusulas: unas hablan de logistica (envio, empaque,
garantia) y otras opinan sobre un aspecto del producto. El sentimiento global lo
decide UNA de las cláusulas de opinion, y las demas actuan como distractores.
Medido sobre train: las palabras de opinion estan repartidas casi 50/50 entre
resenas positivas y negativas, asi que la presencia de "terrible" o "buenisima"
NO indica la clase. Por eso una bolsa de palabras se estanca alrededor de 0,89:
suma las contribuciones de todas las cláusulas y no puede representar "cual
opinion manda".

La solucion se construye en tres pasos, todos aprendidos de los datos:

1. DETECTOR DE CLAUSULAS DE OPINION. Las resenas neutrales del dataset solo
   hablan de logistica, nunca opinan. Sirven entonces como ejemplos negativos
   limpios para entrenar un clasificador de cláusulas "opina / no opina". Los
   ejemplos positivos se obtienen de un detector por vocabulario (las palabras
   que nunca aparecen en resenas neutrales son vocabulario de opinion), que tiene
   precision alta pero se le escapan las opiniones frasales armadas con palabras
   de relleno ("cumple de sobra", "no vale lo que cuesta", "se queda corta"). El
   clasificador generaliza a esas, porque aprende la sintaxis evaluativa y no
   solo el lexico. Medido: la polaridad por cláusula pasa de 0,845 con el
   detector por vocabulario a 0,978 con el aprendido.

2. POLARIDAD POR CLAUSULA. Las resenas con UNA sola cláusula de opinion dan la
   polaridad de esa cláusula sin ambiguedad. Para las demas se usa su ULTIMA
   cláusula de opinion, que coincide con la etiqueta de la resena en el 91% de
   los casos. Etiquetar TODAS las cláusulas con la etiqueta de la resena
   (supervision distante plana) NO funciona: una misma cláusula aparece como
   decisiva en unas resenas y como distractora en otras, y su etiqueta queda
   ~50/50.

3. QUE OPINION MANDA. Las cláusulas de opinion se numeran por recencia (O1 = la
   ultima, O2 = la penultima, ...) y se emiten sus puntuaciones de polaridad
   junto con los marcadores de discurso de la resena. Medido sobre pares en
   conflicto: gana la opinion posterior el 91% de las veces, sube al 97% si lleva
   marcador de resumen ("al final", "con todo") y BAJA al 39% si lleva marcador
   secundario ("de paso", "por cierto"). Esas reglas no se codifican a mano: se
   entregan como rasgos y el clasificador final aprende a pesarlas.
"""
import re
import unicodedata
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

try:
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    _STEMMER = SnowballStemmer('spanish')
    _STOP = set(stopwords.words('spanish'))
except Exception:                                    # pragma: no cover
    _STEMMER = None
    _STOP = set()

# Palabras que NO se eliminan como stop words porque cambian el sentido
PALABRAS_RELEVANTES = {"no", "nunca", "jamás", "sin", "ni", "tampoco", "pero", "aunque",
                       "muy", "bastante", "demasiado", "más", "menos", "poco"}


def quitar_tildes(texto):
    """Quita tildes y convierte la enie en n."""
    descompuesto = unicodedata.normalize('NFD', texto)
    return ''.join(c for c in descompuesto if unicodedata.category(c) != 'Mn')


_RELEVANTES = {quitar_tildes(p) for p in PALABRAS_RELEVANTES}
STOP_SIN_TILDE = {quitar_tildes(p) for p in _STOP} - _RELEVANTES


@lru_cache(maxsize=None)
def raiz(palabra):
    """Stemming con cache: las mismas palabras se repiten miles de veces."""
    return _STEMMER.stem(palabra) if _STEMMER is not None else palabra


# Conectores que abren una nueva cláusula despues de una coma. Se amplia la lista
# del envio 2 con los marcadores de resumen y de concesion que el generador usa
# para senalar cual opinion manda.
CONECTORES = (r'(?:sin embargo|aunque|pero|eso si|aun asi|y no|de paso|a fin de cuentas|'
              r'al final|por otro lado|con todo|en resumen|en fin|la verdad sea dicha|'
              r'ahora que lo pienso|dicho esto|de todas formas|por cierto|ademas)')
PATRON_CORTE = re.compile(r'(?<=[.!?])\s+|\s*;\s*|\s*,\s*(?=' + CONECTORES + r'\b)')

# Marcadores de discurso, agrupados por funcion. Se emiten como token propio para
# que el modelo pueda aprender que "al final" senala la opinion decisiva y que
# "de paso" senala una secundaria.
MARCADORES = {
    'pero': 'MARC_contraste', 'sin embargo': 'MARC_contraste',
    'aunque': 'MARC_concesion', 'eso si': 'MARC_concesion', 'aun asi': 'MARC_concesion',
    'de paso': 'MARC_secundario', 'por cierto': 'MARC_secundario', 'ademas': 'MARC_secundario',
    'a fin de cuentas': 'MARC_resumen', 'al final': 'MARC_resumen',
    'en resumen': 'MARC_resumen', 'en fin': 'MARC_resumen', 'con todo': 'MARC_resumen',
    'de todas formas': 'MARC_resumen',
    'por otro lado': 'MARC_otrolado',
    'la verdad sea dicha': 'MARC_enfasis', 'dicho esto': 'MARC_enfasis',
    'ahora que lo pienso': 'MARC_enfasis',
}
NOMBRES_MARCAS = ('MARC_contraste', 'MARC_concesion', 'MARC_secundario',
                  'MARC_resumen', 'MARC_otrolado', 'MARC_enfasis')
_RE_MARCADORES = re.compile(r'\b(' + '|'.join(sorted(MARCADORES, key=len, reverse=True)) + r')\b')


def separar_clausulas(texto):
    """Minusculas, sin tildes, y corte en oraciones, punto y coma y conectores."""
    texto = quitar_tildes(texto.lower().strip())
    return [c.strip(' .!?,;') for c in PATRON_CORTE.split(texto) if c and c.strip(' .!?,;')]


def tokens_crudos(clausula):
    """Tokens sin stemming ni filtro, para contar vocabulario."""
    return re.sub(r'[^\w\s]', ' ', clausula).split()


def preprocesar_clausula(clausula):
    """Normaliza, tokeniza, quita stop words y aplica stemming."""
    clausula = re.sub(r'[^\w\s]', '', clausula)
    return [raiz(p) for p in clausula.split() if p not in STOP_SIN_TILDE]


def marcadores_de(clausula):
    """Marcadores de discurso presentes en la cláusula."""
    return sorted({MARCADORES[m] for m in _RE_MARCADORES.findall(clausula)})


def inferir_clase_neutral(X, y, min_freq=5):
    """Averigua CUAL de las clases es la neutral, sin depender de su nombre.

    Es necesario porque algunos meta-estimadores de scikit-learn (por ejemplo
    StackingClassifier) codifican las etiquetas a enteros antes de pasarlas a los
    modelos base, asi que no se puede buscar la cadena 'neutral'.

    Criterio: para cada clase se cuentan las palabras frecuentes del corpus que
    NUNCA aparecen en sus resenas. La neutral maximiza ese conteo, porque le
    falta el vocabulario de opinion de AMBAS polaridades, mientras que a las
    positivas solo les falta el negativo y viceversa.
    """
    total, por_clase = Counter(), {}
    for texto, etiqueta in zip(X, list(y)):
        tk = []
        for clausula in separar_clausulas(texto):
            tk += tokens_crudos(clausula)
        total.update(tk)
        por_clase.setdefault(etiqueta, Counter()).update(tk)
    frecuentes = {w for w, n in total.items() if n >= min_freq}
    ausencias = {c: len(frecuentes - set(cnt)) for c, cnt in por_clase.items()}
    return max(ausencias, key=ausencias.get)


# ===========================================================================
#  Detector de cláusulas de opinion
# ===========================================================================
class DetectorOpinion(BaseEstimator):
    """Decide, para cada cláusula, si expresa una opinion o es relleno.

    Se entrena en dos etapas (ver la explicacion del encabezado del modulo):
    primero un detector por vocabulario de alta precision, y con el se etiquetan
    los ejemplos positivos de un clasificador de cláusulas cuyos negativos
    limpios son todas las cláusulas de las resenas neutrales.
    """

    def __init__(self, min_freq_opinion=5, max_freq_neutral=0, umbral=0.0,
                 C_detector=1.0, clase_neutral=None):
        self.min_freq_opinion = min_freq_opinion
        self.max_freq_neutral = max_freq_neutral
        self.umbral = umbral
        self.C_detector = C_detector
        self.clase_neutral = clase_neutral

    def fit(self, X, y):
        X, y = list(X), list(y)
        self.clase_neutral_ = (self.clase_neutral if self.clase_neutral is not None
                               else inferir_clase_neutral(X, y, self.min_freq_opinion))

        # --- etapa A: vocabulario de opinion (semilla de alta precision) ---
        total, neutral = Counter(), Counter()
        for texto, etiqueta in zip(X, y):
            for clausula in separar_clausulas(texto):
                tk = tokens_crudos(clausula)
                total.update(tk)
                if etiqueta == self.clase_neutral_:
                    neutral.update(tk)
        self.vocab_opinion_ = {w for w, n in total.items()
                               if n >= self.min_freq_opinion
                               and neutral[w] <= self.max_freq_neutral}

        # --- etapa B: clasificador de cláusulas ---
        textos, etiquetas = [], []
        for texto, etiqueta in zip(X, y):
            clausulas = separar_clausulas(texto)
            if etiqueta == self.clase_neutral_:
                textos += clausulas
                etiquetas += [0] * len(clausulas)          # negativos limpios
            else:
                for clausula in clausulas:
                    if self._opinion_por_vocabulario(clausula):
                        textos.append(clausula)
                        etiquetas.append(1)                # positivos de la semilla

        self.modelo_ = Pipeline([
            ('rep', FeatureUnion([
                ('pal', TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, min_df=2)),
                ('car', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5),
                                        min_df=2, sublinear_tf=True)),
            ])),
            ('clf', LinearSVC(C=self.C_detector, random_state=42, class_weight='balanced')),
        ]).fit(textos, etiquetas)
        return self

    def _opinion_por_vocabulario(self, clausula):
        return any(w in self.vocab_opinion_ for w in tokens_crudos(clausula))

    def puntuar(self, clausulas):
        """Puntuacion de 'es opinion' para una lista de cláusulas (en un solo lote)."""
        if not clausulas:
            return np.zeros(0)
        return np.asarray(self.modelo_.decision_function(list(clausulas))).ravel()

    def analizar_lote(self, X):
        """Devuelve, por resena, (clausulas, roles).

        Todas las cláusulas de todas las resenas se puntuan en UNA sola llamada,
        que es mucho mas rapido que puntuar resena por resena.
        """
        por_resena, todas = [], []
        for texto in X:
            clausulas = separar_clausulas(texto)
            por_resena.append(clausulas)
            todas += clausulas
        puntos = self.puntuar(todas)

        salida, i = [], 0
        for clausulas in por_resena:
            n = len(clausulas)
            s = puntos[i:i + n]
            i += n
            indices = [k for k in range(n) if s[k] > self.umbral]
            rol = {}
            for orden, k in enumerate(reversed(indices)):      # O1 = la ultima
                rol[k] = f"O{min(orden + 1, 3)}"
            salida.append((clausulas, [rol.get(k, "F") for k in range(n)]))
        return salida


# ===========================================================================
#  Analizador: calcula TODAS las vistas de la resena de una sola pasada
# ===========================================================================
_COLS_TEXTO = ('epl', 'roles', 'ultima', 'o1', 'o1_o2')
COLS_DENSAS = (
    ['pol_o1', 'pol_o2', 'pol_o3', 'pol_o1_menos_o2', 'signo_o1', 'signo_o2',
     'suma', 'media', 'mayor_magnitud', 'n_positivas', 'n_negativas',
     'todas_pos', 'todas_neg', 'n_opinion', 'n_clausulas', 'frac_opinion',
     'sin_opinion', 'ultima_opina', 'score_op_max', 'score_op_min']
    + [f'marca_O1_{m}' for m in NOMBRES_MARCAS]
    + [f'marca_O2_{m}' for m in NOMBRES_MARCAS]
    + [f'pol_o1_x_{m}' for m in NOMBRES_MARCAS]
)


class AnalizadorDeResenas(BaseEstimator, TransformerMixin):
    """Transforma el texto crudo en un DataFrame con todas las vistas de la resena.

    Columnas de texto, para alimentar vectorizadores TF-IDF distintos:
      ``epl``     marcado por posicion del envio 2 (E/P/L), que se conserva
      ``roles``   marcado por recencia de OPINION (O1/O2/O3/F) + marcadores
      ``ultima``  ultima cláusula del texto (bloque de caracteres del envio 2)
      ``o1``      ultima cláusula de OPINION, la candidata a decisiva
      ``o1_o2``   las dos ultimas cláusulas de opinion

    Columnas numericas (``COLS_DENSAS``): polaridad por rol, acuerdos y
    desacuerdos entre cláusulas, conteos y marcadores de discurso.

    Se ajusta UNA sola vez y produce todas las vistas juntas, en lugar de repetir
    el detector en cada rama de un FeatureUnion.
    """

    def __init__(self, min_freq_opinion=5, max_freq_neutral=0, umbral=0.0,
                 C_detector=1.0, C_polaridad=1.0, usar_pseudo=True,
                 usar_marcadores=True, clase_neutral=None):
        self.min_freq_opinion = min_freq_opinion
        self.max_freq_neutral = max_freq_neutral
        self.umbral = umbral
        self.C_detector = C_detector
        self.C_polaridad = C_polaridad
        self.usar_pseudo = usar_pseudo
        self.usar_marcadores = usar_marcadores
        self.clase_neutral = clase_neutral

    # ------------------------------------------------------------------ fit
    def fit(self, X, y=None):
        if y is None:
            raise ValueError("AnalizadorDeResenas necesita las etiquetas en fit: "
                             "aprende de las resenas neutrales que cláusulas son relleno.")
        X, y = list(X), list(y)
        self.det_ = DetectorOpinion(min_freq_opinion=self.min_freq_opinion,
                                    max_freq_neutral=self.max_freq_neutral,
                                    umbral=self.umbral, C_detector=self.C_detector,
                                    clase_neutral=self.clase_neutral).fit(X, y)
        neutral = self.det_.clase_neutral_

        # --- modelo de polaridad por cláusula ---
        analisis = self.det_.analizar_lote(X)
        textos, etiquetas = [], []
        for (clausulas, roles), etiqueta in zip(analisis, y):
            if etiqueta == neutral:
                continue
            ops = [c for c, r in zip(clausulas, roles) if r.startswith('O')]
            if not ops:
                continue
            if len(ops) == 1 or self.usar_pseudo:
                textos.append(ops[-1])            # unica, o la ultima (91% correcta)
                etiquetas.append(etiqueta)

        if len(set(etiquetas)) < 2:
            self.mod_pol_ = None
            return self
        self.mod_pol_ = Pipeline([
            ('rep', FeatureUnion([
                ('pal', TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, min_df=2)),
                ('car', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5),
                                        min_df=2, sublinear_tf=True)),
            ])),
            ('clf', LinearSVC(C=self.C_polaridad, random_state=42)),
        ]).fit(textos, etiquetas)
        # No se fija una orientacion absoluta del signo: decision_function es
        # consistente entre fit y transform y el clasificador final aprende si el
        # positivo queda hacia arriba o hacia abajo.
        return self

    # -------------------------------------------------------------- transform
    def transform(self, X):
        X = list(X)
        analisis = self.det_.analizar_lote(X)

        # polaridad de TODAS las cláusulas de opinion, en un solo lote
        planas, cortes = [], []
        for clausulas, roles in analisis:
            ops = [c for c, r in zip(clausulas, roles) if r.startswith('O')]
            cortes.append(len(ops))
            planas += ops
        if planas and self.mod_pol_ is not None:
            todas = np.asarray(self.mod_pol_.decision_function(planas)).ravel()
        else:
            todas = np.zeros(len(planas))

        # puntuaciones del detector, para saber cuan "opinion" es cada cláusula
        planas_det, cortes_det = [], []
        for clausulas, _ in analisis:
            cortes_det.append(len(clausulas))
            planas_det += clausulas
        det_s = self.det_.puntuar(planas_det)

        filas, i, j = [], 0, 0
        for idx, (clausulas, roles) in enumerate(analisis):
            n_cl = max(len(clausulas), 1)
            n_op = cortes[idx]
            s = todas[i:i + n_op]; i += n_op
            s_det = det_s[j:j + len(clausulas)]; j += len(clausulas)

            ops_marcas = [marcadores_de(c) for c, r in zip(clausulas, roles) if r.startswith('O')]

            o1 = float(s[-1]) if n_op >= 1 else 0.0
            o2 = float(s[-2]) if n_op >= 2 else 0.0
            o3 = float(np.mean(s[:-2])) if n_op > 2 else 0.0
            s_op = s_det[[k for k, r in enumerate(roles) if r.startswith('O')]] if n_op else np.zeros(0)

            fila = {
                'pol_o1': o1, 'pol_o2': o2, 'pol_o3': o3, 'pol_o1_menos_o2': o1 - o2,
                'signo_o1': float(np.sign(o1)), 'signo_o2': float(np.sign(o2)),
                'suma': float(s.sum()) if n_op else 0.0,
                'media': float(s.mean()) if n_op else 0.0,
                'mayor_magnitud': float(s[np.argmax(np.abs(s))]) if n_op else 0.0,
                'n_positivas': float((s > 0).sum()), 'n_negativas': float((s < 0).sum()),
                'todas_pos': float(n_op > 0 and (s > 0).all()),
                'todas_neg': float(n_op > 0 and (s < 0).all()),
                'n_opinion': float(n_op), 'n_clausulas': float(n_cl),
                'frac_opinion': n_op / n_cl, 'sin_opinion': float(n_op == 0),
                'ultima_opina': float(bool(roles) and roles[-1].startswith('O')),
                'score_op_max': float(s_op.max()) if n_op else 0.0,
                'score_op_min': float(s_op.min()) if n_op else 0.0,
            }
            for pos in (1, 2):
                marcas = ops_marcas[-pos] if n_op >= pos else []
                for m in NOMBRES_MARCAS:
                    fila[f'marca_O{pos}_{m}'] = float(m in marcas)
            marcas1 = ops_marcas[-1] if n_op else []
            for m in NOMBRES_MARCAS:
                fila[f'pol_o1_x_{m}'] = o1 * float(m in marcas1)

            # --- vistas de texto ---
            tokens_roles = []
            for clausula, rol in zip(clausulas, roles):
                tokens_roles += [f"{rol}_{t}" for t in preprocesar_clausula(clausula)]
                if self.usar_marcadores:
                    tokens_roles += [f"{rol}_{m}" for m in marcadores_de(clausula)]
            fila['roles'] = " ".join(tokens_roles)

            tokens_epl = []
            n = len(clausulas)
            for k2, clausula in enumerate(clausulas):
                pre = "L" if k2 == n - 1 else ("P" if k2 == n - 2 else "E")
                tokens_epl += [f"{pre}_{t}" for t in preprocesar_clausula(clausula)]
            fila['epl'] = " ".join(tokens_epl)

            fila['ultima'] = re.sub(r'[^\w\s]', '', clausulas[-1]) if clausulas else ""
            ops_txt = [c for c, r in zip(clausulas, roles) if r.startswith('O')]
            if not ops_txt and clausulas:
                ops_txt = [clausulas[-1]]
            fila['o1'] = re.sub(r'[^\w\s]', '', ops_txt[-1]) if ops_txt else ""
            fila['o1_o2'] = re.sub(r'[^\w\s]', '', " ".join(ops_txt[-2:])) if ops_txt else ""

            filas.append(fila)

        return pd.DataFrame(filas, columns=list(_COLS_TEXTO) + list(COLS_DENSAS))


# ===========================================================================
#  Compatibilidad con el envio 2 / envio 3 (para poder comparar)
# ===========================================================================
def marcar_EPL(texto):
    """Marcado del envio 2: E/P/L segun la posicion de la cláusula en el texto."""
    clausulas = separar_clausulas(texto)
    n = len(clausulas)
    tokens = []
    for i, clausula in enumerate(clausulas):
        pre = "L" if i == n - 1 else ("P" if i == n - 2 else "E")
        tokens += [f"{pre}_{t}" for t in preprocesar_clausula(clausula)]
    return " ".join(tokens)


def marcar_EPL_lote(textos):
    return [marcar_EPL(t) for t in textos]


def ultima_clausula_lote(textos):
    salida = []
    for t in textos:
        cl = separar_clausulas(t)
        salida.append(re.sub(r'[^\w\s]', '', cl[-1]) if cl else "")
    return salida
