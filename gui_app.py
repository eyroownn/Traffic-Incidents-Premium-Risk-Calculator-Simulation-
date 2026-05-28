import customtkinter as ctk
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, r2_score, mean_squared_error, mean_absolute_error

class InsuranceApp(ctk.CTk):
    def __init__(self, df, model_manager):
        super().__init__()
        self.df = df
        self.mm = model_manager
        
        self.title("RISK CALCULATOR THROUGH METRO MANILA TRAFFIC ANALYTICS")
        self.geometry("1050x980")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Configure grid weights for responsiveness
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # UI Components
        self.setup_ui()

    def setup_ui(self):
        # Main Container
        main_container = ctk.CTkFrame(self)
        main_container.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        main_container.grid_columnconfigure(0, weight=1)
        main_container.grid_rowconfigure(4, weight=1)
        
        # Header Section
        header_frame = ctk.CTkFrame(main_container, fg_color="#1e1e2f", corner_radius=10)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        header_frame.grid_columnconfigure(0, weight=1)
        
        self.header = ctk.CTkLabel(
            header_frame,
            text="🛡️ Traffic Accident Risk & Premium Analytics",
            font=("Segoe UI", 22, "bold"),
            text_color="#ecf0f1"
        )
        self.header.pack(pady=15)
        
        # Input Section - Given clean layout spacing definitions
        input_frame = ctk.CTkFrame(main_container, fg_color="#14141f", corner_radius=10)
        input_frame.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        input_frame.grid_columnconfigure(0, weight=1)
        
        input_title = ctk.CTkLabel(
            input_frame,
            text="Parameters Matrix Selection",
            font=("Segoe UI", 16, "bold"),
            text_color="#b2bec3"
        )
        input_title.pack(anchor="w", padx=20, pady=(15, 10))
        
        # City Selection
        self._create_input_row(
            input_frame,
            "City Profile",
            "city",
            sorted([c for c in self.df['City'].unique() if pd.notna(c)]),
            "Manila"
        )
        
        # Weather Selection
        self._create_input_row(
            input_frame,
            "Weather Condition",
            "weather",
            ["Clear", "Cloudy", "Rain", "Storm"],
            "Clear"
        )
        
        # Vehicle Type Selection
        self._create_input_row(
            input_frame,
            "Vehicle Classification",
            "vehicle",
            sorted([v for v in self.df['Vehicle_Type'].unique() if pd.notna(v)]),
            "Car"
        )
        
        # Traffic Volume Selection
        self._create_input_row(
            input_frame,
            "Traffic Density",
            "traffic",
            ["Low", "Moderate", "Heavy"],
            "Moderate"
        )
        
        # Button Section
        button_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        button_frame.grid(row=2, column=0, sticky="ew", pady=(0, 15))
        button_frame.grid_columnconfigure((0, 1), weight=1)

        self.calc_btn = ctk.CTkButton(
            button_frame,
            text="⚡ Run Risk Assessment",
            command=self.calculate,
            font=("Segoe UI", 14, "bold"),
            height=45,
            corner_radius=6,
            fg_color="#34495e",
            text_color="#ffffff",
            hover_color="#2c3e50"
        )
        self.calc_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.reset_btn = ctk.CTkButton(
            button_frame,
            text="🔄 Reset Matrix",
            command=self.reset_form,
            font=("Segoe UI", 13),
            height=45,
            corner_radius=6,
            fg_color="#2d3436",
            text_color="#b2bec3",
            hover_color="#636e72"
        )
        self.reset_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        # Tabview for Results and Visualization
        self.tabview = ctk.CTkTabview(main_container, corner_radius=8, border_width=1, border_color="#2d3436")
        self.tabview.grid(row=4, column=0, sticky="nsew", pady=(0, 10))
        self.tabview._segmented_button.configure(font=("Segoe UI", 12))
        
        self.tabview.add("📊 Analysis")
        self.tabview.add("📈 Visualization")
        self.tabview.add("🧠 Model Report")

        # Analysis Tab - Result Display
        analysis_tab = self.tabview.tab("📊 Analysis")
        analysis_tab.grid_rowconfigure(0, weight=1)
        analysis_tab.grid_columnconfigure(0, weight=1)

        self.result_box = ctk.CTkTextbox(
            analysis_tab,
            font=("Consolas", 12),
            corner_radius=6,
            border_width=1,
            border_color="#2d3436"
        )
        self.result_box.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.result_box.insert("1.0", "🔍 Select variables and execute calculation to display telemetry report metrics...\n")
        self.result_box.configure(state="disabled")

        self.graph_label = ctk.CTkLabel(
            analysis_tab,
            text="📊 Visual metrics are updated inside the matching Visualization tab layout frame.",
            font=("Segoe UI", 12),
            text_color="#636e72"
        )
        self.graph_label.grid(row=1, column=0, sticky="w", padx=10, pady=(5, 5))

        self.graph_options = [
            "Average Severity by Weather",
            "Average Severity by Vehicle Type",
            "Average Severity by Traffic Volume",
            "Risk Distribution",
            "Severity Distribution",
            "Damage Cost Distribution"
        ]
        self.selected_graph = self.graph_options[0]
        self.has_calculated = False
        self.visual_canvas_widget = None

        # Visualization Tab Component Design
        visual_tab = self.tabview.tab("📈 Visualization")
        visual_tab.grid_rowconfigure(1, weight=1)
        visual_tab.grid_columnconfigure(0, weight=1)

        viz_control_frame = ctk.CTkFrame(visual_tab, fg_color="#14141f", corner_radius=8)
        viz_control_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 8))
        for idx in range(3):
            viz_control_frame.grid_columnconfigure(idx, weight=1)

        button_texts = self.graph_options
        for idx, graph_name in enumerate(button_texts):
            btn = ctk.CTkButton(
                viz_control_frame,
                text=graph_name,
                command=lambda name=graph_name: self.select_graph(name),
                font=("Segoe UI", 12),
                height=35,
                corner_radius=6,
                fg_color="#2d3436",
                text_color="#dfdfdf",
                hover_color="#4b5563"
            )
            row = idx // 3
            col = idx % 3
            btn.grid(row=row, column=col, sticky="ew", padx=4, pady=4)

        # Center Layout Frame Scroll Container Bounds
        self.visual_canvas_frame = ctk.CTkScrollableFrame(visual_tab, fg_color="#14141f", corner_radius=8, width=720, height=560)
        self.visual_canvas_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.visual_canvas_frame.grid_rowconfigure(0, weight=1)
        self.visual_canvas_frame.grid_columnconfigure(0, weight=1)
        
        # Model Report Workspace
        model_tab = self.tabview.tab("🧠 Model Report")
        model_tab.grid_rowconfigure(0, weight=1)
        model_tab.grid_columnconfigure(0, weight=1)

        self.model_report_box = ctk.CTkTextbox(
            model_tab,
            font=("Consolas", 12),
            corner_radius=6,
            border_width=1,
            border_color="#2d3436"
        )
        self.model_report_box.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.update_model_report_tab()

        # Footer Frame Layout Matrix
        footer_frame = ctk.CTkFrame(main_container, fg_color="#14141f", corner_radius=10)
        footer_frame.grid(row=5, column=0, sticky="ew")
        footer_frame.grid_columnconfigure(0, weight=1)
        
        footer_text = ctk.CTkLabel(
            footer_frame,
            text=f"Last Evaluation Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Target Classification System Accuracy: {self.mm.accuracy:.2%}",
            font=("Segoe UI", 11),
            text_color="#636e72"
        )
        footer_text.pack(pady=8)

    def _create_input_row(self, parent, label_text, var_name, values, default):
        """Creates explicitly expanded inputs and drop downs for enhanced visibility."""
        row_frame = ctk.CTkFrame(parent, fg_color="transparent")
        row_frame.pack(fill="x", padx=20, pady=12) 
        row_frame.grid_columnconfigure(1, weight=1)
        
        label = ctk.CTkLabel(
            row_frame,
            text=label_text,
            font=("Segoe UI", 14, "bold"), 
            text_color="#dfdfdf",
            width=180,
            anchor="w"
        )
        label.grid(row=0, column=0, sticky="w", padx=(0, 10))
        
        var = ctk.StringVar(value=default)
        setattr(self, f"{var_name}_var", var)
        var.trace_add("write", self._on_selection_change)
        
        dropdown = ctk.CTkComboBox(
            row_frame,
            values=values,
            variable=var,
            font=("Helvetica", 14), 
            height=38, 
            state="readonly",
            corner_radius=4,
            border_width=1,
            border_color="#2d3436"
        )
        dropdown.grid(row=0, column=1, sticky="ew")

    def update_model_report_tab(self):
        """Processes statistical errors and formats report logs into the UI display."""
        try:
            self.model_report_box.configure(state="normal")
            self.model_report_box.delete("1.0", "end")
            
            features_list = ['City', 'Weather_Condition', 'Vehicle_Type', 'Traffic_Volume']
            X = pd.get_dummies(self.df[features_list])
            y = self.df['High_Risk_Target']
            X_encoded = X.reindex(columns=self.mm.training_columns, fill_value=0)
            
            y_pred = self.mm.model.predict(X_encoded)
            y_prob = self.mm.model.predict_proba(X_encoded)[:, 1]
            
            # Regression Metrics Evaluation Vector
            f1 = f1_score(y, y_pred, average='weighted')
            r2 = r2_score(y, y_prob)
            rmse = np.sqrt(mean_squared_error(y, y_prob))
            mae = mean_absolute_error(y, y_prob)
            
            metrics_display = f"""{'='*60}
📊 ADVANCED MODEL PERFORMANCE METRICS MATRIX
{'='*60}

📈 CLASSIFICATION INTEGRITY LOGS:
   System Accuracy Index:       {self.mm.accuracy:.2%}
   Weighted F1-Score:           {f1:.4f}

  ERROR VARIANCE & REGRESSION COEFFICIENTS (Risk Probabilities):
   R-Squared (R² Metric):       {r2:.4f}
   Root Mean Squared Error:     {rmse:.4f}
   Mean Absolute Error (MAE):   {mae:.4f}

{'-'*60}
📋 BASE DETAILED CLASSIFICATION FRAME:
{self.mm.metrics_report}
{'='*60}"""
            self.model_report_box.insert("1.0", metrics_display)
            self.model_report_box.configure(state="disabled")
        except Exception as e:
            self.model_report_box.configure(state="normal")
            self.model_report_box.insert("1.0", f"Error rendering structural metrics logs: {str(e)}")
            self.model_report_box.configure(state="disabled")

    def _on_selection_change(self, *args):
        if not getattr(self, 'has_calculated', False):
            return
        self.generate_graphs(self.city_var.get(), self.selected_graph)

    def select_graph(self, graph_type):
        self.selected_graph = graph_type
        if getattr(self, 'has_calculated', False):
            self.generate_graphs(self.city_var.get(), self.selected_graph)

    def calculate(self):
        try:
            prob = self.mm.predict_risk(
                self.city_var.get(),
                self.weather_var.get(),
                self.vehicle_var.get(),
                self.traffic_var.get()
            )
            
            base_fee = 2500.00
            city_mean_cost = self.df[self.df['City'] == self.city_var.get()]['Damage_Cost_PHP'].mean()
            if pd.isna(city_mean_cost):
                city_mean_cost = 0.0
            final_premium = max(8500, base_fee + (prob * city_mean_cost * 0.05))
            
            if prob > 0.5:
                risk_tier = "🔴 HIGH RISK PROFILE"
            elif prob > 0.2:
                risk_tier = "🟡 MODERATE RISK PROFILE"
            else:
                risk_tier = "🟢 LOW RISK PROFILE"
            
            result_text = f"""
{'='*60}
         🛡️  TRAFFIC RISK ASSESSMENT REPORT SUMMARY  🛡️
{'='*60}

📍 LOCATION SCENARIO:  {self.city_var.get().upper()}
🌦️ ATMOSPHERIC METRIC: {self.weather_var.get()}
🚗 CLASSIFICATION:     {self.vehicle_var.get()}
🚦 DENSITY VALUE:      {self.traffic_var.get()}

{'-'*60}

⚠️  RISK TELEMETRY:
   Risk Evaluation:      {risk_tier}
   Calculated Variance:  {prob:.2%}
   
💰 FINANCIAL FORECAST ANALYSIS:
   Mean Damage Cost (Regional Index): PHP {city_mean_cost:,.2f}
   
✅ FINAL ESTIMATED PREMIUM MARGIN:    PHP {final_premium:,.2f}

{'-'*60}
📊 Engine Confidence Matrix: {self.mm.accuracy:.2%}
⏱️  Calculated Processing Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}
"""
            
            self.result_box.configure(state="normal")
            self.result_box.delete("1.0", "end")
            self.result_box.insert("1.0", result_text)
            self.result_box.configure(state="disabled")
            
            self.update_model_report_tab()
            self.has_calculated = True
            self.generate_graphs(self.city_var.get(), self.selected_graph)
            
        except Exception as e:
            self.result_box.configure(state="normal")
            self.result_box.delete("1.0", "end")
            self.result_box.insert("1.0", f"❌ Error: {str(e)}\n\nVerify matrix vector selections.")
            self.result_box.configure(state="disabled")

    def generate_graphs(self, city, graph_type=None):
        """Builds a compact, centered, screenshot-ready square dashboard plot."""
        try:
            if graph_type is None:
                graph_type = self.selected_graph if hasattr(self, 'selected_graph') else self.graph_options[0]

            city_data = self.df[self.df['City'] == city]
            weather_sel = self.weather_var.get() if hasattr(self, 'weather_var') else None
            vehicle_sel = self.vehicle_var.get() if hasattr(self, 'vehicle_var') else None
            traffic_sel = self.traffic_var.get() if hasattr(self, 'traffic_var') else None

            subset_mask = pd.Series(True, index=city_data.index)
            if weather_sel:
                subset_mask &= (city_data['Weather_Condition'] == weather_sel)
            if vehicle_sel:
                subset_mask &= (city_data['Vehicle_Type'] == vehicle_sel)
            if traffic_sel:
                subset_mask &= (city_data['Traffic_Volume'] == traffic_sel)

            subset_data = city_data[subset_mask]
            
            # --- SMALL SCREENSHOT-READY SQUARE CONFIGURATION ---
            fig = Figure(figsize=(3.6, 3.6), dpi=115, facecolor='#14141f', edgecolor='#2d3436')
            fig.patch.set_facecolor('#14141f')
            ax = fig.add_subplot(1, 1, 1)
            ax.set_facecolor('#1e1e2f') 
            
            ax.tick_params(axis='x', colors='#b2bec3', labelsize=8.5)
            ax.tick_params(axis='y', colors='#b2bec3', labelsize=8.5)
            colors = ['#54a0ff', '#ff6b6b', '#feca57', '#1dd1a1']

            if city_data.empty:
                ax.text(0.5, 0.5, 'No statistical values reported.', color='#b2bec3', ha='center', va='center', fontsize=10)
            
            elif graph_type == 'Average Severity by Weather':
                weather_risk = city_data.groupby('Weather_Condition')['Severity'].mean().sort_values(ascending=False)
                if not weather_risk.empty:
                    bars = weather_risk.plot(kind='bar', ax=ax, color=colors[0], edgecolor='#14141f', linewidth=1, width=0.45)
                    ax.set_xlabel('Weather Profile', fontsize=9, color='#b2bec3', labelpad=5)
                    ax.set_ylabel('Mean Severity Weight', fontsize=9, color='#b2bec3', labelpad=5)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
                    ax.grid(axis='y', alpha=0.05, color='white')
                    
                    for bar in bars.patches:
                        yval = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.04, f"{yval:.2f}", 
                                ha='center', va='bottom', color='#dfdfdf', fontsize=7.5, fontweight='bold')
            
            elif graph_type == 'Average Severity by Vehicle Type':
                vehicle_risk = city_data.groupby('Vehicle_Type')['Severity'].mean().sort_values(ascending=False)
                if not vehicle_risk.empty:
                    bars = vehicle_risk.plot(kind='bar', ax=ax, color=colors[1], edgecolor='#14141f', linewidth=1, width=0.45)
                    ax.set_xlabel('Vehicle Configuration', fontsize=9, color='#b2bec3', labelpad=5)
                    ax.set_ylabel('Mean Severity Weight', fontsize=9, color='#b2bec3', labelpad=5)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
                    ax.grid(axis='y', alpha=0.05, color='white')
                    
                    for bar in bars.patches:
                        yval = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.04, f"{yval:.2f}", 
                                ha='center', va='bottom', color='#dfdfdf', fontsize=7.5, fontweight='bold')
            
            elif graph_type == 'Average Severity by Traffic Volume':
                traffic_risk = city_data.groupby('Traffic_Volume')['Severity'].mean().sort_values(ascending=False)
                if not traffic_risk.empty:
                    bars = traffic_risk.plot(kind='bar', ax=ax, color=colors[2], edgecolor='#14141f', linewidth=1, width=0.35)
                    ax.set_xlabel('Density Threshold', fontsize=9, color='#b2bec3', labelpad=5)
                    ax.set_ylabel('Mean Severity Weight', fontsize=9, color='#b2bec3', labelpad=5)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                    ax.grid(axis='y', alpha=0.05, color='white')
                    
                    for bar in bars.patches:
                        yval = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.04, f"{yval:.2f}", 
                                ha='center', va='bottom', color='#dfdfdf', fontsize=7.5, fontweight='bold')
            
            elif graph_type == 'Risk Distribution':
                risk_counts = (city_data['Severity'] > 3).value_counts()
                if risk_counts.sum() > 0:
                    labels = ['High Risk' if idx == True else 'Low Risk' for idx in risk_counts.index]
                    wedges, texts, autotexts = ax.pie(
                        risk_counts, labels=labels, autopct='%1.1f%%', 
                        colors=[colors[1], colors[3]], startangle=90, pctdistance=0.65,
                        textprops={'color': 'white', 'fontsize': 8.5, 'fontweight': 'bold'}
                    )
                    centre_circle = plt.Circle((0,0), 0.50, fc='#1e1e2f', edgecolor='#2d3436', linewidth=1)
                    ax.add_artist(centre_circle)
                    for text in texts:
                        text.set_color('#b2bec3')
                        text.set_fontsize(9)
            
            elif graph_type == 'Severity Distribution':
                severity_city = city_data['Severity'].dropna()
                severity_subset = subset_data['Severity'].dropna()
                if len(severity_city) > 0:
                    bins = np.histogram_bin_edges(severity_city, bins=10)
                    ax.hist(severity_city, bins=bins, density=True, alpha=0.2, color=colors[0], edgecolor='#54a0ff', label='City Matrix')
                    if len(severity_subset) > 0:
                        ax.hist(severity_subset, bins=bins, density=True, alpha=0.5, color=colors[1], edgecolor='#ff6b6b', label='Scenario Filter')
                    
                    mu_c, sigma_c = severity_city.mean(), severity_city.std()
                    x = np.linspace(severity_city.min(), severity_city.max(), 200)
                    if sigma_c > 0:
                        ax.plot(x, stats.norm.pdf(x, mu_c, sigma_c), color='#54a0ff', linewidth=1.5, label='Base Density')
                    
                    ax.set_xlabel('Incident Magnitude', fontsize=9, color='#b2bec3', labelpad=5)
                    ax.set_ylabel('Density Scalar', fontsize=9, color='#b2bec3', labelpad=5)
                    legend = ax.legend(fontsize=7.5, facecolor='#14141f', edgecolor='#2d3436')
                    plt.setp(legend.get_texts(), color='#b2bec3')
            
            elif graph_type == 'Damage Cost Distribution':
                damage_city = city_data['Damage_Cost_PHP'].dropna()
                damage_subset = subset_data['Damage_Cost_PHP'].dropna()
                if len(damage_city) > 0:
                    bins_d = np.histogram_bin_edges(damage_city, bins=12)
                    ax.hist(damage_city, bins=bins_d, density=True, alpha=0.2, color=colors[2], edgecolor='#feca57', label='City Base')
                    if len(damage_subset) > 0:
                        ax.hist(damage_subset, bins=bins_d, density=True, alpha=0.5, color=colors[1], edgecolor='#ff6b6b', label='Scenario Filter')
                    
                    mu_d_c, sigma_d_c = damage_city.mean(), damage_city.std()
                    x_damage = np.linspace(damage_city.min(), damage_city.max(), 200)
                    if sigma_d_c > 0:
                        ax.plot(x_damage, stats.norm.pdf(x_damage, mu_d_c, sigma_d_c), color='#feca57', linewidth=1.5, label='Base Deviation')
                    
                    ax.set_xlabel('Loss (PHP)', fontsize=9, color='#b2bec3', labelpad=5)
                    ax.set_ylabel('Density Scalar', fontsize=9, color='#b2bec3', labelpad=5)
                    legend = ax.legend(fontsize=7.5, facecolor='#14141f', edgecolor='#2d3436')
                    plt.setp(legend.get_texts(), color='#b2bec3')
                    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

            ax.set_title(f'{graph_type.upper()}', fontsize=10.5, fontweight='bold', color='#ecf0f1', pad=10)
            fig.tight_layout()

            if self.visual_canvas_widget:
                try:
                    self.visual_canvas_widget.destroy()
                except Exception:
                    pass
                self.visual_canvas_widget = None

            self.visual_canvas_frame.grid_rowconfigure(0, weight=1)
            self.visual_canvas_frame.grid_columnconfigure(0, weight=1)

            canvas = FigureCanvasTkAgg(fig, master=self.visual_canvas_frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            self.visual_canvas_widget = widget
            
            # Explicit padding parameters to center-align the square plot area perfectly
            widget.grid(row=0, column=0, sticky="", padx=40, pady=40)
            self.last_fig = fig
            
        except Exception as e:
            print(f"Graph display execution exception: {e}")

    def open_visualization_window(self):
        try:
            if getattr(self, 'last_fig', None) is None:
                try:
                    self.generate_graphs(self.city_var.get())
                except Exception:
                    pass

            fig = getattr(self, 'last_fig', None)
            if fig is None:
                popup = ctk.CTkToplevel(self)
                popup.title("Visualization Dashboard")
                popup.geometry("400x200")
                lbl = ctk.CTkLabel(popup, text="Perform initial analysis calculation sequence.", font=("Segoe UI", 12))
                lbl.pack(expand=True, padx=20, pady=20)
                return

            if getattr(self, 'viz_window', None) is not None and self.viz_window.winfo_exists():
                self.viz_window.lift()
                self.viz_window.focus_force()
                if getattr(self, 'viz_canvas_widget', None):
                    self.viz_canvas_widget.destroy()
                if getattr(self, 'viz_scroll_frame', None) is None:
                    self.viz_scroll_frame = ctk.CTkScrollableFrame(master=self.viz_window, fg_color="transparent")
                    self.viz_scroll_frame.pack(fill="both", expand=True)
                canvas = FigureCanvasTkAgg(fig, master=self.viz_scroll_frame)
                canvas.draw()
                widget = canvas.get_tk_widget()
                self.viz_canvas_widget = widget
                widget.pack(fill="both", expand=True, padx=10, pady=10)
                return

            win = ctk.CTkToplevel(self)
            win.title(f"Telemetry Metric Frame - {self.city_var.get()}")
            win.geometry("500x500") 
            win.rowconfigure(0, weight=1)
            win.columnconfigure(0, weight=1)
            self.viz_window = win

            scroll = ctk.CTkScrollableFrame(master=win, fg_color="transparent")
            scroll.pack(fill="both", expand=True)
            self.viz_scroll_frame = scroll

            canvas = FigureCanvasTkAgg(fig, master=scroll)
            canvas.draw()
            widget = canvas.get_tk_widget()
            self.viz_canvas_widget = widget
            widget.pack(fill="both", expand=True, padx=10, pady=10)
            return
        except Exception as e:
            print(f"Top-level frame interface exception: {e}")

    def reset_form(self):
        self.city_var.set("Manila")
        self.weather_var.set("Clear")
        self.vehicle_var.set("Car")
        self.traffic_var.set("Moderate")
        
        self.result_box.configure(state="normal")
        self.result_box.delete("1.0", "end")
        self.result_box.insert("1.0", "🔍 Select variables and execute calculation to display telemetry report metrics...\n")
        self.result_box.configure(state="disabled")
        
        if self.visual_canvas_widget:
            self.visual_canvas_widget.destroy()
            self.visual_canvas_widget = None
        self.has_calculated = False
        if hasattr(self, 'last_fig'):
            self.last_fig = None

        if hasattr(self, 'graph_label'):
            self.graph_label.configure(text="📊 Visual metrics are updated inside the matching Visualization tab layout frame.")