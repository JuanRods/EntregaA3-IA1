from flask import Flask, request, jsonify, render_template, send_from_directory
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import StandardScaler

import pandas as pd
import unicodedata
import numpy as np
import re
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE_PATH = os.path.join(BASE_DIR, "html", "user")

CSV_PATH = os.path.join(BASE_DIR, "database", "jogos_limpos.csv")

STATIC_PATH = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_PATH, exist_ok=True)

app = Flask(
    __name__,
    template_folder=TEMPLATE_PATH,
    static_folder=STATIC_PATH
)

print("📁 BASE_DIR      =", BASE_DIR)
print("📁 TEMPLATE_PATH =", TEMPLATE_PATH)
print("📁 CSV_PATH      =", CSV_PATH)
print("📁 STATIC_PATH   =", STATIC_PATH)


@app.route("/css/<path:filename>")
def css_files(filename):
    return send_from_directory(os.path.join(BASE_DIR, "css"), filename)


@app.route("/js/<path:filename>")
def js_files(filename):
    return send_from_directory(os.path.join(BASE_DIR, "js"), filename)


@app.route("/img/<path:filename>")
def img_files(filename):
    return send_from_directory(os.path.join(BASE_DIR, "img"), filename)


def normalizar(txt):
    if pd.isna(txt):
        return ""
    txt = unicodedata.normalize("NFD", str(txt))
    txt = "".join(c for c in txt if unicodedata.category(c) != "Mn")
    return txt.lower().strip()


def carregar_csv():
    if not os.path.exists(CSV_PATH):
        raise RuntimeError(f"CSV NÃO ENCONTRADO: {CSV_PATH}")

    try:
        df = pd.read_csv(CSV_PATH, encoding="utf-8")
    except pd.errors.ParserError as e:
        print(f"⚠ ParserError ao ler CSV com o engine padrão: {e}\n")
        print("👉 Tentando novamente ignorando linhas problemáticas (on_bad_lines='skip')...")
        df = pd.read_csv(
            CSV_PATH,
            encoding="utf-8",
            engine="python",
            on_bad_lines="skip"
        )

    print(f"✅ CSV carregado: {len(df)} linhas (após limpar linhas problemáticas)")

    df.fillna("", inplace=True)

    if "generos_full" not in df.columns:
        df["generos_full"] = df.get("genero", "").astype(str)

    if "plataformas_full" not in df.columns:
        df["plataformas_full"] = df.get("plataforma", "").astype(str)

    if "tags_recomendacao" not in df.columns:
        df["tags_recomendacao"] = ""

    for col in ["nota", "popularidade", "ano"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if df["nota"].max() != df["nota"].min():
        df["nota_norm"] = (df["nota"] - df["nota"].min()) / (df["nota"].max() - df["nota"].min())
    else:
        df["nota_norm"] = 0.0

    if df["popularidade"].max() != df["popularidade"].min():
        df["pop_norm"] = (df["popularidade"] - df["popularidade"].min()) / (df["popularidade"].max() - df["popularidade"].min())
    else:
        df["pop_norm"] = 0.0

    df["score_alvo"] = 0.6 * df["nota_norm"] + 0.4 * df["pop_norm"]

    df["nome_norm"] = df["nome"].map(normalizar)
    df["genero_norm"] = df.get("genero", "").map(normalizar)
    df["publisher_norm"] = df["publisher"].map(normalizar)
    df["plataforma_norm"] = df.get("plataforma", "").map(normalizar)
    df["palavras_chave_norm"] = df.get("palavras_chave", "").map(normalizar)
    df["descricao_norm"] = df.get("descricao", "").map(normalizar)

    df["generos_full_norm"] = df["generos_full"].map(normalizar)
    df["plataformas_full_norm"] = df["plataformas_full"].map(normalizar)
    df["tags_recomendacao_norm"] = df["tags_recomendacao"].map(normalizar)

    return df


try:
    df = carregar_csv()
except Exception as e:
    print("\n❌ ERRO AO CARREGAR CSV:", e)
    raise e

rf_model = None
nn_model = None

feature_cols = None

rf_metrics = {}
nn_metrics = {}

rf_cls_metrics = {}
nn_cls_metrics = {}

nn_scaler_x = None
nn_scaler_y = None


def preparar_features(df_src: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df_src.index)

    X["popularidade"] = df_src["popularidade"].astype(float)
    X["ano"] = df_src["ano"].astype(float)
    X["nota"] = df_src["nota"].astype(float)

    gen_text = df_src.get("generos_full_norm", df_src.get("genero_norm", ""))
    tag_text = df_src.get("tags_recomendacao_norm", "")
    plat_text = df_src.get("plataformas_full_norm", df_src.get("plataforma_norm", ""))

    genre_tokens = [
        "acao", "shooter", "rpg", "aventura", "estrategia", "plataforma",
        "puzzle", "indie", "terror", "mundo aberto", "mmo",
        "esportes", "corrida", "simulacao", "casual", "familia"
    ]

    for g in genre_tokens:
        col_name = f"gen_{g.replace(' ', '_')}"
        X[col_name] = gen_text.str.contains(g, na=False).astype(int)

    tag_map = {
        "mundo_aberto": "mundo aberto exploracao aventura",
        "terror": "terror zumbi horror",
        "coop": "multiplayer co-op cooperativo online",
        "pvp": "competitivo ranked pvp",
        "fantasia": "fantasia medieval magia dragao",
        "puzzle": "puzzle quebra-cabeca enigma",
        "familia": "familia leve relaxante kids",
        "indie": "indie pixel 2d metroidvania",
    }

    for key, phrase in tag_map.items():
        X[f"tag_{key}"] = tag_text.str.contains(phrase, na=False).astype(int)

    plat_patterns = {
        "pc": r"pc|windows|steam",
        "playstation": r"playstation|ps4|ps5|ps3|ps2|ps vita",
        "xbox": r"xbox",
        "nintendo": r"nintendo|switch|wii|3ds",
        "mobile": r"android|ios|mobile|celular",
    }

    for label, pattern in plat_patterns.items():
        X[f"plat_{label}"] = plat_text.str.contains(pattern, regex=True, na=False).astype(int)

    return X


def preparar_base_treino_teste():
    X = preparar_features(df)
    y = df["score_alvo"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def gerar_graficos_model(slug: str, titulo: str, metrics: dict):
    train_mse = metrics.get("mse_treino", 0)
    test_mse = metrics.get("mse_teste", 0)
    train_r2 = metrics.get("r2_treino", 0)
    test_r2 = metrics.get("r2_teste", 0)
    train_mae = metrics.get("mae_treino", 0)
    test_mae = metrics.get("mae_teste", 0)

    plt.figure()
    plt.title(f"{titulo} - MSE (Treino x Teste)")
    plt.bar(["Treino", "Teste"], [train_mse, test_mse])
    plt.ylabel("Erro quadrático médio")
    mse_path = os.path.join(STATIC_PATH, f"{slug}_mse.png")
    plt.tight_layout()
    plt.savefig(mse_path)
    plt.close()
    print(f"📊 [{titulo}] Gráfico MSE salvo em {mse_path}")

    plt.figure()
    plt.title(f"{titulo} - R² (Treino x Teste)")
    plt.bar(["Treino", "Teste"], [train_r2, test_r2])
    plt.ylabel("R²")
    r2_path = os.path.join(STATIC_PATH, f"{slug}_r2.png")
    plt.tight_layout()
    plt.savefig(r2_path)
    plt.close()
    print(f"📊 [{titulo}] Gráfico R² salvo em {r2_path}")

    plt.figure()
    plt.title(f"{titulo} - MAE (Treino x Teste)")
    plt.bar(["Treino", "Teste"], [train_mae, test_mae])
    plt.ylabel("Erro absoluto médio")
    mae_path = os.path.join(STATIC_PATH, f"{slug}_mae.png")
    plt.tight_layout()
    plt.savefig(mae_path)
    plt.close()
    print(f"📊 [{titulo}] Gráfico MAE salvo em {mae_path}")


def gerar_matriz_confusao_fig(slug: str, titulo: str, cm: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{titulo} - Matriz de Confusão")

    fig.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    ax.set_xlabel("Predito pelo modelo")
    ax.set_ylabel("Valor real")

    siglas = [["VN", "FP"],
              ["FN", "VP"]]

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                f"{siglas[i][j]}\n{cm[i,j]}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="black"
            )

    legenda_texto = (
        "Legenda:\n"
        "VP – Verdadeiro Positivo\n"
        "VN – Verdadeiro Negativo\n"
        "FP – Falso Positivo\n"
        "FN – Falso Negativo\n"
        "Classes: 0 = Não recomendar, 1 = Recomendar"
    )

    fig.text(
        0.01, -0.08,
        legenda_texto,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="gray",
            alpha=0.95
        )
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    path = os.path.join(STATIC_PATH, f"{slug}_cm.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"📊 [{titulo}] Matriz de confusão salva em {path}")


def gerar_roc_curve(slug: str, titulo: str, y_true_bin, scores):
    fpr, tpr, _ = roc_curve(y_true_bin, scores)
    auc_val = roc_auc_score(y_true_bin, scores)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("Taxa de falsos positivos")
    plt.ylabel("Taxa de verdadeiros positivos")
    plt.title(f"{titulo} - Curva ROC")
    plt.legend(loc="lower right")
    path = os.path.join(STATIC_PATH, f"{slug}_roc.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"📊 [{titulo}] Curva ROC salva em {path}")


def gerar_comparacao_modelos_cls(rf_cls: dict, nn_cls: dict):
    if not rf_cls or not nn_cls:
        return

    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    rf_vals = [
        rf_cls.get("accuracy", 0),
        rf_cls.get("precision", 0),
        rf_cls.get("recall", 0),
        rf_cls.get("f1", 0),
    ]
    nn_vals = [
        nn_cls.get("accuracy", 0),
        nn_cls.get("precision", 0),
        nn_cls.get("recall", 0),
        nn_cls.get("f1", 0),
    ]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure()
    plt.title("Comparação de métricas de classificação (RF vs NN)")
    plt.bar(x - width/2, rf_vals, width, label="Random Forest")
    plt.bar(x + width/2, nn_vals, width, label="Rede Neural")
    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Pontuação")
    plt.legend()
    path = os.path.join(STATIC_PATH, "comparacao_modelos_cls.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"📊 Gráfico de comparação RF vs NN salvo em {path}")


def treinar_modelos():
    global rf_model, nn_model, feature_cols
    global rf_metrics, nn_metrics
    global rf_cls_metrics, nn_cls_metrics
    global nn_scaler_x, nn_scaler_y

    X_train, X_test, y_train, y_test, cols = preparar_base_treino_teste()
    feature_cols = cols

    print("\n🌳 Treinando RandomForestRegressor...")
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    rf_train_pred = rf.predict(X_train)
    rf_test_pred = rf.predict(X_test)

    rf_metrics = {
        "n_treino": len(X_train),
        "n_teste": len(X_test),
        "r2_treino": float(rf.score(X_train, y_train)),
        "r2_teste": float(rf.score(X_test, y_test)),
        "mse_treino": float(mean_squared_error(y_train, rf_train_pred)),
        "mse_teste": float(mean_squared_error(y_test, rf_test_pred)),
        "mae_treino": float(mean_absolute_error(y_train, rf_train_pred)),
        "mae_teste": float(mean_absolute_error(y_test, rf_test_pred)),
    }

    rf_model = rf
    print("✅ Random Forest treinado.")
    gerar_graficos_model("rf", "Random Forest", rf_metrics)

    limiar = np.quantile(y_test, 0.75)
    print(f"\n🔎 Limiar de recomendação (top 25% score_alvo) = {limiar:.4f}")

    y_test_bin = (y_test >= limiar).astype(int)
    rf_test_bin = (rf_test_pred >= limiar).astype(int)

    cm_rf = confusion_matrix(y_test_bin, rf_test_bin)
    print("\n📊 Matriz de confusão - Random Forest (0 = não recomendar, 1 = recomendar)")
    print(cm_rf)
    print("\n📄 Classification report - RF:")
    print(classification_report(y_test_bin, rf_test_bin, digits=4))

    rf_cls_metrics = {
        "accuracy": float(accuracy_score(y_test_bin, rf_test_bin)),
        "precision": float(precision_score(y_test_bin, rf_test_bin, zero_division=0)),
        "recall": float(recall_score(y_test_bin, rf_test_bin, zero_division=0)),
        "f1": float(f1_score(y_test_bin, rf_test_bin, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_bin, rf_test_pred)),
    }

    gerar_matriz_confusao_fig("rf", "Random Forest", cm_rf)
    gerar_roc_curve("rf", "Random Forest", y_test_bin, rf_test_pred)

    print("\n🤖 Treinando Rede Neural (MLPRegressor)...")

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

    nn = MLPRegressor(
        hidden_layer_sizes=(9, 9),
        max_iter=1000,
        random_state=42
    )
    nn.fit(X_train_scaled, y_train_scaled)

    nn_train_pred_scaled = nn.predict(X_train_scaled)
    nn_test_pred_scaled = nn.predict(X_test_scaled)

    nn_train_pred = scaler_y.inverse_transform(nn_train_pred_scaled.reshape(-1, 1)).ravel()
    nn_test_pred = scaler_y.inverse_transform(nn_test_pred_scaled.reshape(-1, 1)).ravel()

    nn_metrics = {
        "n_treino": len(X_train),
        "n_teste": len(X_test),
        "r2_treino": float(1 - mean_squared_error(y_train, nn_train_pred) / np.var(y_train)),
        "r2_teste": float(1 - mean_squared_error(y_test, nn_test_pred) / np.var(y_test)),
        "mse_treino": float(mean_squared_error(y_train, nn_train_pred)),
        "mse_teste": float(mean_squared_error(y_test, nn_test_pred)),
        "mae_treino": float(mean_absolute_error(y_train, nn_train_pred)),
        "mae_teste": float(mean_absolute_error(y_test, nn_test_pred)),
    }

    nn_model = nn
    nn_scaler_x = scaler_x
    nn_scaler_y = scaler_y
    print("✅ Rede Neural treinada.")
    gerar_graficos_model("nn", "Rede Neural (MLP)", nn_metrics)

    nn_test_bin = (nn_test_pred >= limiar).astype(int)
    cm_nn = confusion_matrix(y_test_bin, nn_test_bin)
    print("\n📊 Matriz de confusão - Rede Neural (0 = não recomendar, 1 = recomendar)")
    print(cm_nn)
    print("\n📄 Classification report - NN:")
    print(classification_report(y_test_bin, nn_test_bin, digits=4))

    nn_cls_metrics = {
        "accuracy": float(accuracy_score(y_test_bin, nn_test_bin)),
        "precision": float(precision_score(y_test_bin, nn_test_bin, zero_division=0)),
        "recall": float(recall_score(y_test_bin, nn_test_bin, zero_division=0)),
        "f1": float(f1_score(y_test_bin, nn_test_bin, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_bin, nn_test_pred)),
    }

    gerar_matriz_confusao_fig("nn", "Rede Neural (MLP)", cm_nn)
    gerar_roc_curve("nn", "Rede Neural (MLP)", y_test_bin, nn_test_pred)

    gerar_comparacao_modelos_cls(rf_cls_metrics, nn_cls_metrics)

    imprimir_resumo_modelos()


def imprimir_resumo_modelos():
    print("\n" + "=" * 70)
    print("RESUMO GERAL DOS TESTES DOS MODELOS (valores para o relatório)")
    print("=" * 70)

    def bloco(nome, m, cls=None):
        if not m:
            print(f"\n- {nome}: modelo ainda não treinado.\n")
            return
        print(f"\n- {nome}:")
        print(f"  • Amostras treino: {m.get('n_treino', '?')}")
        print(f"  • Amostras teste:  {m.get('n_teste', '?')}")
        print(f"  • R² treino:       {m.get('r2_treino', 0):.4f}")
        print(f"  • R² teste:        {m.get('r2_teste', 0):.4f}")
        print(f"  • MSE treino:      {m.get('mse_treino', 0):.6f}")
        print(f"  • MSE teste:       {m.get('mse_teste', 0):.6f}")
        print(f"  • MAE treino:      {m.get('mae_treino', 0):.6f}")
        print(f"  • MAE teste:       {m.get('mae_teste', 0):.6f}")
        if cls:
            print(f"  • Accuracy (binário): {cls.get('accuracy', 0):.4f}")
            print(f"  • Precision (binário): {cls.get('precision', 0):.4f}")
            print(f"  • Recall (binário):    {cls.get('recall', 0):.4f}")
            print(f"  • F1-score (binário):  {cls.get('f1', 0):.4f}")
            print(f"  • ROC AUC:             {cls.get('roc_auc', 0):.4f}")

    bloco("Random Forest Regressor", rf_metrics, rf_cls_metrics)
    bloco("Rede Neural (MLPRegressor)", nn_metrics, nn_cls_metrics)

    print("\n" + "=" * 70 + "\n")


treinar_modelos()


def extrair_tag(msg_norm, nome_tag):
    m = re.search(rf"\[{nome_tag.lower()}=([^\]]+)\]", msg_norm)
    return m.group(1).strip() if m else None


def extrair_preferencias(msg_norm: str) -> dict:
    genero_tag = extrair_tag(msg_norm, "GENERO")
    humor_tag = extrair_tag(msg_norm, "HUMOR")
    plataforma_tag = extrair_tag(msg_norm, "PLATAFORMA")
    faixa_tag = extrair_tag(msg_norm, "FAIXA")
    nota_tag = extrair_tag(msg_norm, "NOTA")
    ano_tag = extrair_tag(msg_norm, "ANO")
    tags_extra_tag = extrair_tag(msg_norm, "TAGS")

    genero_base = "acao"
    base_txt = (genero_tag or humor_tag or "")

    if any(t in base_txt for t in ["shooter", "fps", "tiro", "arma"]):
        genero_base = "shooter"
    elif any(t in base_txt for t in ["rpg", "fantasia", "medieval", "mago", "dragao", "dragão", "elfo"]):
        genero_base = "rpg"
    elif any(t in base_txt for t in ["estrategia", "estratégia", "tatico", "tática", "civilization"]):
        genero_base = "estrategia"
    elif any(t in base_txt for t in ["aventura", "historia", "história", "narrativa", "exploracao", "exploração"]):
        genero_base = "aventura"
    elif any(t in base_txt for t in ["arcade", "plataforma", "platformer"]):
        genero_base = "plataforma"
    elif any(t in base_txt for t in ["terror", "horror", "zumbi"]):
        genero_base = "terror"
    elif "indie" in base_txt:
        genero_base = "indie"

    if (genero_tag in (None, "", "qualquer")) and genero_base == "acao":
        if any(t in msg_norm for t in ["shooter", "fps", "tiro", "arma"]):
            genero_base = "shooter"
        elif any(t in msg_norm for t in ["rpg", "fantasia", "medieval", "mago", "dragao", "dragão", "elfo"]):
            genero_base = "rpg"
        elif any(t in msg_norm for t in ["estrategia", "estratégia", "tatico", "tática", "civilization"]):
            genero_base = "estrategia"
        elif any(t in msg_norm for t in ["aventura", "historia", "história", "narrativa", "exploracao", "exploração"]):
            genero_base = "aventura"
        elif any(t in msg_norm for t in ["arcade", "plataforma", "platformer"]):
            genero_base = "plataforma"
        elif any(t in msg_norm for t in ["terror", "horror", "zumbi"]):
            genero_base = "terror"
        elif "indie" in msg_norm:
            genero_base = "indie"

    texto_total = msg_norm + " " + (tags_extra_tag or "")

    quer_terror = any(t in texto_total for t in ["terror", "horror", "zumbi"])
    quer_mundo_aberto = ("mundo aberto" in texto_total) or ("open world" in texto_total)
    quer_competitivo = any(t in texto_total for t in ["competitivo", "ranked", "pvp"])
    quer_multiplayer = any(t in texto_total for t in ["multiplayer", "online", "coop", "co-op", "cooperativo"])
    quer_puzzle = any(t in texto_total for t in ["puzzle", "quebra-cabeca", "quebra cabeca", "quebra cabeça", "enigma"])
    quer_familia = any(t in texto_total for t in ["familia", "família", "kids", "infantil", "leve", "relaxante"])
    quer_fantasia = any(t in texto_total for t in ["fantasia", "medieval", "magia", "mago", "dragao", "dragão"])
    quer_historia_forte = any(t in texto_total for t in ["historia forte", "história forte", "narrativa", "emocionante", "drama"])
    quer_indie = "indie" in texto_total or "metroidvania" in texto_total
    quer_adulto = any(t in texto_total for t in ["adulto", "violento", "gore"])

    plataformas = []
    if plataforma_tag and plataforma_tag != "qualquer":
        val = plataforma_tag
        if "pc" in val:
            plataformas.append("pc")
        elif "playstation" in val or "ps" in val:
            plataformas.append("playstation")
        elif "xbox" in val:
            plataformas.append("xbox")
        elif "nintendo" in val or "switch" in val:
            plataformas.append("nintendo")
        elif "mobile" in val or "android" in val or "ios" in val or "celular" in val:
            plataformas.append("mobile")
    else:
        if "pc" in msg_norm or "computador" in msg_norm or "steam" in msg_norm:
            plataformas.append("pc")
        if any(t in msg_norm for t in ["playstation", "ps4", "ps5", "ps3", "ps2", "ps vita", "psvita"]):
            plataformas.append("playstation")
        if "xbox" in msg_norm:
            plataformas.append("xbox")
        if any(t in msg_norm for t in ["nintendo", "switch", "wii", "3ds"]):
            plataformas.append("nintendo")
        if any(t in msg_norm for t in ["android", "ios", "mobile", "celular"]):
            plataformas.append("mobile")

    publishers_keywords = {
        "rockstar": "rockstar games",
        "bethesda": "bethesda softworks",
        "ubisoft": "ubisoft entertainment",
        "ea": "electronic arts",
        "square enix": "square enix",
        "valve": "valve",
        "nintendo": "nintendo",
        "sony": "sony computer entertainment",
        "microsoft": "microsoft studios",
        "konami": "konami",
        "bandai": "bandai namco entertainment",
        "warner": "warner bros. interactive",
        "devolver": "devolver digital",
    }
    publisher = None
    for key, val in publishers_keywords.items():
        if key in msg_norm:
            publisher = val
            break

    nota_min = None
    if nota_tag and nota_tag != "qualquer":
        try:
            nota_min = float(nota_tag.replace(",", "."))
        except Exception:
            nota_min = None

    ano = None
    ano_min = None
    ano_max = None

    if ano_tag and ano_tag != "qualquer":
        if re.fullmatch(r"\d{4}", ano_tag):
            ano = int(ano_tag)
        elif ano_tag == "recentes":
            ano_min = 2019
        elif ano_tag == "antes2015":
            ano_max = 2014
        elif ano_tag == "antes2010":
            ano_max = 2009
    else:
        m_ano = re.search(r"\b(19\d{2}|20\d{2})\b", msg_norm)
        if m_ano:
            ano = int(m_ano.group(1))

    faixa_explicit = None
    if faixa_tag and faixa_tag != "qualquer":
        faixa_explicit = faixa_tag.upper()
    else:
        fx_match = re.search(r"\+(\d{1,2})", msg_norm)
        if fx_match:
            faixa_explicit = f"+{fx_match.group(1)}"
        elif "livre" in msg_norm or "para todos" in msg_norm:
            faixa_explicit = "LIVRE"

    return {
        "genero_base": genero_base,
        "quer_terror": quer_terror,
        "quer_mundo_aberto": quer_mundo_aberto,
        "quer_competitivo": quer_competitivo,
        "quer_multiplayer": quer_multiplayer,
        "quer_puzzle": quer_puzzle,
        "quer_familia": quer_familia,
        "quer_fantasia": quer_fantasia,
        "quer_historia_forte": quer_historia_forte,
        "quer_indie": quer_indie,
        "quer_adulto": quer_adulto,
        "plataformas": plataformas,
        "publisher": publisher,
        "nota_min": nota_min,
        "ano": ano,
        "ano_min": ano_min,
        "ano_max": ano_max,
        "faixa_explicit": faixa_explicit,
    }


def recomendar(msg_original: str):
    msg_norm = normalizar(msg_original)
    print("\n🔎 Requisição recebida:", msg_original)

    genero_tag = extrair_tag(msg_norm, "GENERO") or ""
    tags_tag = extrair_tag(msg_norm, "TAGS") or ""
    prefs = extrair_preferencias(msg_norm)

    caminho = []
    relaxamentos = []

    df_f = df.copy()

    gen_full = df_f.get("generos_full_norm", df_f.get("genero_norm"))
    plat_full = df_f.get("plataformas_full_norm", df_f.get("plataforma_norm"))
    tags_full = df_f.get("tags_recomendacao_norm", df_f["descricao_norm"])

    genero = prefs["genero_base"]

    if genero_tag and genero_tag not in ("qualquer", "nao definido", "não definido"):
        genero = genero_tag

    caminho.append(f"Gênero preferido: {genero}")

    if genero == "shooter":
        mask = gen_full.str.contains("shooter", na=False)
    elif genero == "rpg":
        mask = gen_full.str.contains("rpg", na=False)
    elif genero in ("estrategia", "strategy"):
        mask = gen_full.str.contains("estrategia", na=False)
    elif genero == "aventura":
        mask = gen_full.str.contains("aventura", na=False)
    elif genero in ("plataforma", "arcade"):
        mask = gen_full.str.contains("plataforma|arcade", regex=True, na=False)
    elif genero == "terror":
        mask = gen_full.str.contains("terror", na=False)
    elif genero == "indie":
        mask = gen_full.str.contains("indie", na=False)
    elif genero in ("acao", "action"):
        mask = gen_full.str.contains("acao", na=False)
    else:
        mask = gen_full.str.contains(genero, na=False)

    df_tmp = df_f[mask]
    if not df_tmp.empty:
        df_f = df_tmp
        caminho.append("Filtro de gênero aplicado sobre generos_full.")
    else:
        relaxamentos.append("Gênero: não encontramos jogos exatamente nesse gênero; usamos o catálogo completo.")
        caminho.append("Filtro de gênero removido (sem resultados).")

    def aplica_subfiltro(descricao, mask_local):
        nonlocal df_f
        df_sub = df_f[mask_local]
        if not df_sub.empty:
            df_f = df_sub
            caminho.append(f"Subfiltro aplicado: {descricao}.")
        else:
            relaxamentos.append(f"{descricao.capitalize()}: nenhum jogo disponível com esse recorte; mantivemos os demais filtros.")
            caminho.append(f"Subfiltro removido: {descricao} (sem resultados).")

    if tags_tag and tags_tag not in ("nenhuma", "qualquer"):
        mask_local = tags_full.str.contains(tags_tag, na=False)
        aplica_subfiltro("característica extra (tags selecionadas)", mask_local)

    if prefs["quer_terror"]:
        mask_local = (
            tags_full.str.contains("terror zumbi horror", na=False) |
            gen_full.str.contains("terror", na=False) |
            df_f["descricao_norm"].str.contains("terror|horror|zumbi", regex=True, na=False)
        )
        aplica_subfiltro("jogos com clima de terror", mask_local)

    if prefs["quer_historia_forte"]:
        mask_local = df_f["descricao_norm"].str.contains("historia|história|narrativa|drama|emocionante", regex=True, na=False)
        aplica_subfiltro("foco em história/narrativa", mask_local)

    if prefs["quer_puzzle"]:
        mask_local = (
            tags_full.str.contains("puzzle quebra-cabeca enigma", na=False) |
            gen_full.str.contains("puzzle", na=False)
        )
        aplica_subfiltro("jogos de quebra-cabeça/puzzle", mask_local)

    if prefs["quer_mundo_aberto"]:
        mask_local = (
            tags_full.str.contains("mundo aberto exploracao aventura", na=False) |
            gen_full.str.contains("mundo aberto", na=False) |
            df_f["descricao_norm"].str.contains("mundo aberto|open world|sandbox|exploracao|exploração",
                                                regex=True, na=False)
        )
        aplica_subfiltro("mundo aberto/exploração", mask_local)

    if prefs["quer_multiplayer"]:
        mask_local = (
            tags_full.str.contains("multiplayer co-op cooperativo online", na=False) |
            df_f["descricao_norm"].str.contains("multiplayer|online|cooperativo|coop|co-op", regex=True, na=False)
        )
        aplica_subfiltro("multiplayer/co-op", mask_local)

    if prefs["quer_competitivo"]:
        mask_local = (
            tags_full.str.contains("competitivo ranked pvp", na=False) |
            df_f["descricao_norm"].str.contains("pvp|ranked|competitiv", regex=True, na=False)
        )
        aplica_subfiltro("PVP/competitivo", mask_local)

    if prefs["quer_familia"]:
        mask_local = (
            tags_full.str.contains("familia leve relaxante kids", na=False) |
            df_f["descricao_norm"].str.contains("familia|família|casual|leve|relaxante|kids|infantil",
                                                regex=True, na=False)
        )
        aplica_subfiltro("foco em família/casual", mask_local)

    if prefs["quer_indie"]:
        mask_local = (
            tags_full.str.contains("indie pixel 2d metroidvania", na=False) |
            gen_full.str.contains("indie", na=False)
        )
        aplica_subfiltro("jogos indie/alternativos", mask_local)

    faixa_explicit = prefs["faixa_explicit"]
    quer_adulto = prefs["quer_adulto"]

    if faixa_explicit:
        if faixa_explicit.upper() == "LIVRE":
            mask_fx = df_f["faixa_etaria"].astype(str).isin(["LIVRE", "+10", "+13", "N/A"])
            df_tmp = df_f[mask_fx]
            if not df_tmp.empty:
                df_f = df_tmp
                caminho.append("Faixa etária: jogos livres/para todos.")
            else:
                relaxamentos.append("Faixa etária: não achamos jogos adequados para 'LIVRE'; mantivemos jogos de outras faixas.")
                caminho.append("Filtro de faixa etária removido (sem resultados).")
        else:
            mask_fx = df_f["faixa_etaria"].astype(str) == faixa_explicit
            df_tmp = df_f[mask_fx]
            if not df_tmp.empty:
                df_f = df_tmp
                caminho.append(f"Faixa etária: {faixa_explicit}.")
            else:
                relaxamentos.append(f"Faixa etária {faixa_explicit}: nenhum jogo nessa faixa; mantivemos outras faixas.")
                caminho.append("Filtro de faixa etária removido (sem resultados).")
    else:
        if quer_adulto:
            mask_fx = df_f["faixa_etaria"].astype(str).isin(["+17", "+18", "18", "18+"])
            df_tmp = df_f[mask_fx]
            if not df_tmp.empty:
                df_f = df_tmp
                caminho.append("Faixa etária inferida: adulto (+17/+18).")
            else:
                relaxamentos.append("Jogos adultos: nenhum jogo explicitamente adulto encontrado; mantivemos outras faixas.")
                caminho.append("Filtro adulto removido (sem resultados).")

    plataformas = prefs["plataformas"]
    if plataformas:
        caminho.append("Plataformas escolhidas: " + ", ".join(plataformas))

        mask_total = None
        for p in set(plataformas):
            if p == "pc":
                mask_p = plat_full.str.contains("pc|windows|steam", regex=True, na=False)
            elif p == "playstation":
                mask_p = plat_full.str.contains("playstation|ps4|ps5|ps3|ps2|ps vita", regex=True, na=False)
            elif p == "xbox":
                mask_p = plat_full.str.contains("xbox", regex=True, na=False)
            elif p == "nintendo":
                mask_p = plat_full.str.contains("nintendo|switch|wii|3ds", regex=True, na=False)
            elif p == "mobile":
                mask_p = plat_full.str.contains("android|ios|mobile|celular", regex=True, na=False)
            else:
                mask_p = None

            if mask_p is not None:
                mask_total = mask_p if mask_total is None else (mask_total | mask_p)

        if mask_total is not None:
            df_tmp = df_f[mask_total]
            if not df_tmp.empty:
                df_f = df_tmp
                caminho.append("Filtro de plataforma aplicado usando plataformas_full.")
            else:
                relaxamentos.append("Plataforma: não encontramos jogos na(s) plataforma(s) escolhida(s); mostramos jogos de outras plataformas com o mesmo perfil.")
                caminho.append("Filtro de plataforma removido (sem resultados).")

    publisher = prefs["publisher"]
    if publisher:
        mask_pub = df_f["publisher_norm"].str.contains(normalizar(publisher), na=False)
        df_tmp = df_f[mask_pub]
        if not df_tmp.empty:
            df_f = df_tmp
            caminho.append(f"Publisher preferido: {publisher}.")
        else:
            relaxamentos.append(f"Publisher: nenhum jogo do publisher '{publisher}' no recorte atual; mantivemos outros publishers.")
            caminho.append("Filtro de publisher removido (sem resultados).")

    nota_min = prefs["nota_min"]
    if nota_min is not None:
        df_tmp = df_f[df_f["nota"] >= nota_min]
        if not df_tmp.empty:
            df_f = df_tmp
            caminho.append(f"Nota mínima aplicada: {nota_min}.")
        else:
            relaxamentos.append(f"Nota mínima: nenhum jogo com nota ≥ {nota_min}; mantivemos jogos com notas menores.")
            caminho.append("Filtro de nota mínima removido (sem resultados).")

    ano = prefs["ano"]
    ano_min = prefs["ano_min"]
    ano_max = prefs["ano_max"]

    if ano:
        df_tmp = df_f[df_f["ano"] == ano]
        if not df_tmp.empty:
            df_f = df_tmp
            caminho.append(f"Ano específico: {ano}.")
        else:
            relaxamentos.append(f"Ano: nenhum jogo exatamente de {ano}; mostramos jogos de outros anos.")
            caminho.append("Filtro de ano específico removido (sem resultados).")
    else:
        if ano_min is not None:
            df_tmp = df_f[df_f["ano"] >= ano_min]
            if not df_tmp.empty:
                df_f = df_tmp
                caminho.append(f"Ano mínimo aplicado: {ano_min}.")
            else:
                relaxamentos.append(f"Ano mínimo {ano_min}: nenhum jogo recente o suficiente; mantivemos jogos mais antigos.")
                caminho.append("Filtro de ano mínimo removido (sem resultados).")

        if ano_max is not None:
            df_tmp = df_f[df_f["ano"] <= ano_max]
            if not df_tmp.empty:
                df_f = df_tmp
                caminho.append(f"Ano máximo aplicado: {ano_max}.")
            else:
                relaxamentos.append(f"Ano máximo {ano_max}: nenhum jogo tão antigo; mantivemos jogos mais novos.")
                caminho.append("Filtro de ano máximo removido (sem resultados).")

    if df_f.empty:
        return [{
            "intro": (
                "⚠ Não encontramos nenhum jogo compatível com a combinação de filtros selecionada.\n"
                "Tente ajustar gênero, plataforma ou ano para ver mais resultados."
            )
        }]

    df_f = df_f.copy()
    df_f["score_manual"] = 0.6 * df_f["nota_norm"] + 0.4 * df_f["pop_norm"]

    if rf_model is not None and feature_cols is not None:
        X_cand = preparar_features(df_f)[feature_cols]
        df_f["rf_score"] = rf_model.predict(X_cand)
    else:
        df_f["rf_score"] = 0.0

    if nn_model is not None and nn_scaler_x is not None and nn_scaler_y is not None and feature_cols is not None:
        X_cand = preparar_features(df_f)[feature_cols]
        X_cand_scaled = nn_scaler_x.transform(X_cand)
        nn_pred_scaled = nn_model.predict(X_cand_scaled)
        nn_pred = nn_scaler_y.inverse_transform(nn_pred_scaled.reshape(-1, 1)).ravel()
        df_f["nn_score"] = nn_pred
    else:
        df_f["nn_score"] = 0.0

    df_f["score_final"] = (
        0.3 * df_f["score_manual"] +
        0.4 * df_f["rf_score"] +
        0.3 * df_f["nn_score"]
    )

    df_f = df_f.sort_values("score_final", ascending=False).head(10)

    intro_parts = []
    intro_parts.append("<strong>🎮 Recomendações geradas com base nas suas respostas</strong>")

    if relaxamentos:
        intro_parts.append(
            "<br><br>⚠ <strong>Não encontramos jogos que atendam exatamente a TODOS os filtros.</strong>"
        )
        intro_parts.append("<br>Os seguintes filtros foram <b>flexibilizados</b>:")
        intro_parts.append("<ul style='margin-top:4px; margin-bottom:8px;'>")
        for r in relaxamentos:
            intro_parts.append(f"<li>{r}</li>")
        intro_parts.append("</ul>")
        intro_parts.append(
            "<p style='margin-top:4px;'>➡ Mesmo assim, mantivemos o máximo possível "
            "das suas preferências (gênero, estilo, plataforma e tags extras).</p>"
        )
    else:
        intro_parts.append(
            "<br><br>✅ <strong>Todos os filtros que você escolheu foram respeitados.</strong>"
        )

    intro_parts.append("<br><br><strong>🧠 Caminho da árvore de decisão</strong>")
    intro_parts.append("<ul style='margin-top:4px;'>")
    for passo in caminho:
        intro_parts.append(f"<li>{passo}</li>")
    intro_parts.append("</ul>")

    intro_texto = "".join(intro_parts)

    resposta = [{"intro": intro_texto}]

    for _, row in df_f.iterrows():
        resposta.append({
            "nome": row["nome"],
            "ano": int(row["ano"]) if row["ano"] else 0,
            "genero": row.get("genero", ""),
            "plataforma": row.get("plataforma", ""),
            "publisher": row["publisher"],
            "nota": float(row["nota"]),
            "faixa_etaria": row["faixa_etaria"],
            "descricao": row["descricao"],
            "score": float(row["score_final"])
        })

    return resposta


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        msg = data.get("mensagem", "")

        resp = recomendar(msg)
        return jsonify({"resposta": resp})
    except Exception as e:
        print("❌ ERRO NO /chat:", e)
        return jsonify({"erro": str(e)}), 500


@app.route("/chat_test")
def chat_test():
    exemplo = "quero um rpg de fantasia em mundo aberto estilo skyrim para playstation com nota acima de 4"
    return jsonify({"resposta": recomendar(exemplo)})


@app.route("/model_metrics")
def model_metrics_route():
    return jsonify({
        "random_forest": {
            "regression": rf_metrics,
            "classification": rf_cls_metrics
        },
        "neural_network": {
            "regression": nn_metrics,
            "classification": nn_cls_metrics
        }
    })


if __name__ == "__main__":
    print("\n🔥 Servidor iniciado em: http://localhost:5000/")
    app.run(debug=True)
