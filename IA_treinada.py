import glob # automatiza o processo de pegar todos os arquivos xls na pasta do note
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# ===============================
# 1) Carregar e Preparar Dados
# ===============================

'''
procura todos os arquivos de dados xls na pasta e cria uma lista com os nomes desses arquivos
'''
caminho_pasta = "C:\\Users\\giova\\OneDrive\\Desktop\\FIAP\\dados IA GOODWE"
arquivos = glob.glob(os.path.join(caminho_pasta, "*.xls")) + glob.glob(os.path.join(caminho_pasta, "*.xlsx"))


'''
para cada arquivo:
    le os dados
    padroniza os nomes das colunas 
    converte coluna de tempo pra formato data-hora
    renomeia a coluna de consumo pra load
    limpa dados ruins faltando
    guarda os dados limpos numa lista
    mostra 
'''
lista_dfs = []

for arq in arquivos:
    try:
        if arq.endswith(".xlsx"):
            df_temp = pd.read_excel(arq, header=2, engine="openpyxl")
        else:
            df_temp = pd.read_excel(arq, header=2, engine="xlrd")

        df_temp.columns = df_temp.columns.str.strip().str.lower()

        if "time" in df_temp.columns:
            df_temp["time"] = pd.to_datetime(df_temp["time"], errors="coerce", dayfirst=True)

        for col in df_temp.columns:
            if "load" in col:
                df_temp.rename(columns={col: "load"}, inplace=True)

        df_temp = df_temp.dropna(subset=["time", "load"])
        df_temp["load"] = pd.to_numeric(df_temp["load"], errors="coerce")

        lista_dfs.append(df_temp)
        print(f"✅ {os.path.basename(arq)}: {len(df_temp)} registros")

    except Exception as e:
        print(f"⚠️ Erro em {arq}: {e}") # hahha

df = pd.concat(lista_dfs, ignore_index=True)
df = df.sort_values('time').reset_index(drop=True)

print(f"\n📊 Dataset consolidado: {len(df)} registros de {df['time'].dt.date.nunique()} dias")



# Supondo que df_dia já tem as colunas ["time", "load"]
df["hora"] = df["time"].dt.hour
standby = df[df["hora"].between(0, 5)]["load"].mean()
standby_limite = df[df["hora"].between(0,5)]["load"].quantile(0.9)

# Criação do target
df["deve_desligar"] = ((df["hora"].between(0, 5)) & (df["load"] <= 400)).astype(int)

# Features e target
X = df[["hora", "load"]]
y = df["deve_desligar"]

# Treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# Teste de acurácia
print("Acurácia:", modelo.score(X_test, y_test))

# Simulação de decisão
# Exemplo: hora=2, load=105
previsao = modelo.predict([[2, 105]])
print("Ação:", "Desligar standby!" if previsao[0] == 1 else "Manter ligado!")

df["acao_predita"] = modelo.predict(df[["hora", "load"]])
df["acao_predita"] = df["acao_predita"].map({1: "Desligar standby!", 0: "Manter ligado!"})

print(df[["time", "load", "hora", "acao_predita"]].head(2000))  # Mostra as 20 primeiras decisões
