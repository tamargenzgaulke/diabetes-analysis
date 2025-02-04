import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Carregar o dataset
dataset_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv(dataset_url, names=columns)


# Exibir as primeiras linhas
display(df.head())

# Informações básicas do dataset
display(df.info())

# Estatísticas descritivas
display(df.describe())

# Verificar valores nulos
print("Valores nulos:")
print(df.isnull().sum())

# Visualizar a distribuição da variável alvo
sns.countplot(x='Outcome', data=df)
plt.title('Distribuição de Casos de Diabetes')
plt.show()

# Correlação entre as variáveis
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlação entre as Variáveis')
plt.show()

# Separar variáveis independentes e dependentes
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Treinar modelo de Regressão Logística
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Avaliação do modelo
print("Acurácia da Regressão Logística:", accuracy_score(y_test, y_pred_log))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred_log))
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred_log))

# Treinar modelo de Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Avaliação do modelo Random Forest
print("Acurácia do Random Forest:", accuracy_score(y_test, y_pred_rf))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred_rf))
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred_rf))

