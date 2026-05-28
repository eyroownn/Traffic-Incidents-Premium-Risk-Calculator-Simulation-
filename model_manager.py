import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

class AccidentModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced',
            max_depth=10,
            min_samples_leaf=10
        )
        self.training_columns = []
        self.metrics_report = ""
        self.accuracy = 0.0

    def train(self, df):
        features_list = ['City', 'Weather_Condition', 'Vehicle_Type', 'Traffic_Volume']
        X = pd.get_dummies(df[features_list])
        y = df['High_Risk_Target']
        
        self.training_columns = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        # Calculate Metrics
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.metrics_report = classification_report(y_test, y_pred)
        
        return self.metrics_report, self.accuracy

    def predict_risk(self, city, weather, vehicle, traffic):
        # Prepare input for prediction
        input_data = pd.DataFrame([[city, weather, vehicle, traffic]], 
                                   columns=['City', 'Weather_Condition', 'Vehicle_Type', 'Traffic_Volume'])
        input_encoded = pd.get_dummies(input_data)
        input_scenario = input_encoded.reindex(columns=self.training_columns, fill_value=0)
        
        prob = self.model.predict_proba(input_scenario)[0][1]
        return prob