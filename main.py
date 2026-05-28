from data_manager import load_and_clean_data
from model_manager import AccidentModel
from gui_app import InsuranceApp

def main():
    print("Initializing Application...")
    
    # 1. Load Data
    df = load_and_clean_data('Comprehensive_Traffic_Data_2025.csv')
    
    # 2. Initialize and Train Model
    mm = AccidentModel()
    report, accuracy = mm.train(df)
    
    print("\n--- Model Classification Report ---")
    print(report)
    print(f"Accuracy: {accuracy:.2%}\n")
    
    # 3. Launch UI
    app = InsuranceApp(df, mm)
    app.mainloop()

if __name__ == "__main__":
    main()