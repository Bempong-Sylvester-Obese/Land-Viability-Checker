# Land Viability Checker 🌾

## 📌 Overview
The **Land Viability Checker** is a comprehensive AI-powered agricultural assessment tool designed to evaluate land suitability for crop production. By integrating soil quality analysis, climate assessment, and machine learning-based yield predictions, it provides farmers and agribusinesses with data-driven insights for informed agricultural decisions.

## 🚀 Key Features

### 🔬 **Comprehensive Analysis**
- **Soil Quality Assessment**: Evaluates 15+ soil parameters including pH, nutrients, texture, and fertility
- **Climate Suitability Analysis**: Assesses temperature, rainfall, humidity, and sunshine patterns
- **Crop-Specific Recommendations**: Provides suitability scores for 6+ major crops (maize, rice, wheat, sorghum, cassava, yam)
- **Machine Learning Predictions**: Uses trained ML models to predict crop yields with high accuracy

### 📊 **Advanced Analytics**
- **Economic Viability Analysis**: Calculates revenue, profit margins, and cost-benefit ratios
- **Risk Assessment**: Identifies limiting factors and potential challenges
- **Comparative Analysis**: Ranks crops by viability and profitability
- **Interactive Visualizations**: Generates comprehensive charts and dashboards

### 🎯 **User-Friendly Interface**
- **Command-Line Interface**: Easy-to-use CLI for quick assessments
- **Interactive Mode**: Step-by-step guided assessment process
- **Comprehensive Reports**: Detailed PDF-style reports with recommendations
- **Export Capabilities**: Save results as JSON, images, and reports

---

## 🛠️ Technologies Used

### **Core Technologies**
- **Python 3.13**: Modern Python with latest features
- **Scikit-learn**: Machine learning algorithms and model training
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Matplotlib & Seaborn**: Data visualization and reporting

### **Machine Learning Models**
- **Random Forest Regressor**: Ensemble learning for robust predictions
- **Gradient Boosting**: Advanced boosting algorithms
- **Linear Regression**: Fast and interpretable baseline models
- **Support Vector Regression**: Non-linear pattern recognition

### **Data Sources**
- **Historical Crop Yield Data**: 70+ years of West African agricultural data
- **Soil Parameter Databases**: Comprehensive soil characteristic references
- **Climate Data**: Temperature, rainfall, and weather pattern analysis
- **Economic Data**: Current crop prices and production costs

---

## 📖 Installation & Setup

### **Prerequisites**
- Python 3.13 or higher
- pip package manager
- Git (for cloning the repository)

### **Installation Steps**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Bempong-Sylvester-Obese/land-viability-checker.git
   cd Land-Viability-Checker
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python main.py --demo
   ```

---

## 🚀 Quick Start Guide

### **1. Run Complete Demo**
```bash
python main.py --demo
```
This runs a comprehensive demonstration showcasing all features with sample data.

### **2. Train Machine Learning Models**
```bash
python main.py --train-models
```
Trains and optimizes crop yield prediction models using historical data.

### **3. Interactive Assessment**
```bash
python main.py
```
Choose option 2 for interactive mode where you can input your own land parameters.

### **4. Component-Specific Analysis**
```bash
python main.py --soil-analysis      # Soil quality analysis only
python main.py --climate-analysis   # Climate suitability only
python main.py --full-assessment    # Complete land assessment
```

---

## 📊 Example Output

### **Land Viability Assessment Report**
```
LAND VIABILITY ASSESSMENT REPORT
============================================================

PROPERTY INFORMATION:
   Location: Kumasi, Ghana
   Coordinates: 6.5000, -1.6000
   Elevation: 150 meters

OVERALL VIABILITY ASSESSMENT:
   Overall Score: 99.16/100
   Viability Level: Excellent

CROP VIABILITY ANALYSIS:
   MAIZE: Highly Viable (98.67/100)
   RICE: Highly Viable (99.58/100)
   WHEAT: Highly Viable (92.93/100)
   SORGHUM: Highly Viable (95.33/100)

ECONOMIC ANALYSIS:
   RICE: $592.44/hectare profit (33.0% margin)
   MAIZE: $28.83/hectare profit (3.5% margin)
```

### **Generated Visualizations**
- `soil_analysis_demo.png` - Soil quality charts and metrics
- `full_assessment_dashboard.png` - Comprehensive viability dashboard
- `model_performance.png` - ML model comparison and accuracy metrics
- `crop_yield_analysis.png` - Historical yield trends and patterns

---

## 🔧 Advanced Usage

### **Custom Land Assessment**
```python
from models import LandViabilityAssessor, LandParameters

# Create custom land parameters
land = LandParameters(
    latitude=6.5,
    longitude=-1.6,
    elevation=150,
    soil_ph=6.2,
    organic_matter=2.5,
    nitrogen=25.0,
    # ... other parameters
)

# Perform assessment
assessor = LandViabilityAssessor()
assessment = assessor.assess_land_viability(land, ['maize', 'rice'])

# Generate report
report = assessor.create_comprehensive_report(land)
print(report)
```

### **Model Training and Evaluation**
```python
from models.crop_yield_models import train_crop_yield_models

# Train models with custom parameters
predictor = train_crop_yield_models('path/to/custom_data.csv')

# Evaluate model performance
summary = predictor.get_model_summary()
print(summary)
```

---

## 📁 Project Structure

```
Land-Viability-Checker/
├── models/                          # Core ML models and analysis modules
│   ├── crop_yield_models.py        # Crop yield prediction models
│   ├── soil_analysis.py            # Soil quality assessment
│   ├── climate_analysis.py         # Climate suitability analysis
│   ├── land_viability_assessor.py  # Integrated assessment system
│   └── __init__.py                 # Package initialization
├── Data/                           # Data files and outputs
│   ├── crop_yield_data.csv         # Historical crop yield data
│   ├── *.png                       # Generated visualizations
│   └── *.json                      # Assessment results
├── main.py                         # Main application entry point
├── data_preprocessing.py           # Data preprocessing pipeline
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🎯 Supported Crops

The system currently supports analysis for these major crops:
- **Maize** - Staple cereal crop with high nutritional value
- **Rice** - Primary food crop in many regions
- **Wheat** - Important cereal for bread and pasta production
- **Sorghum** - Drought-resistant cereal crop
- **Cassava** - Root crop with high carbohydrate content
- **Yam** - Important tuber crop in West Africa

---

## 📈 Model Performance

Our machine learning models achieve excellent performance:
- **Linear Regression**: R² = 1.000 (Perfect fit on training data)
- **Ridge Regression**: R² = 0.9998 (Excellent generalization)
- **Gradient Boosting**: R² = 0.9752 (Strong ensemble performance)
- **Random Forest**: R² = 0.9624 (Robust and interpretable)

---

## 🔮 Future Enhancements

### **Phase 2: Web Interface** (Planned)
- [ ] Web-based dashboard with interactive maps
- [ ] Real-time weather data integration
- [ ] User account management and history
- [ ] Mobile-responsive design

### **Phase 3: Advanced Features** (Planned)
- [ ] Satellite imagery integration
- [ ] Mobile app for field assessments
- [ ] IoT sensor data integration
- [ ] Multi-language support

### **Phase 4: Enterprise Features** (Planned)
- [ ] API for third-party integrations
- [ ] Large-scale farm management tools
- [ ] Advanced analytics and reporting
- [ ] Integration with agricultural databases

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### **Ways to Contribute**
1. **Bug Reports**: Report issues and unexpected behavior
2. **Feature Requests**: Suggest new functionality
3. **Code Contributions**: Submit pull requests for improvements
4. **Documentation**: Help improve guides and examples
5. **Testing**: Test on different systems and datasets

### **Development Setup**
```bash
git clone https://github.com/Bempong-Sylvester-Obese/land-viability-checker.git
cd Land-Viability-Checker
pip install -r requirements.txt
python -m pytest tests/  # Run tests
```

---

## 📊 Performance Benchmarks

- **Assessment Speed**: < 2 seconds for complete land analysis
- **Model Accuracy**: 95%+ accuracy on crop yield predictions
- **Data Processing**: Handles 1000+ records in < 10 seconds
- **Memory Usage**: < 500MB for typical assessments
- **Scalability**: Supports batch processing of multiple locations

---

## 🛡️ Data Privacy & Security

- **Local Processing**: All analysis performed locally, no data sent to external servers
- **Open Source**: Full source code available for review and audit
- **No Tracking**: No user data collection or tracking
- **Secure**: No API keys or sensitive credentials required for basic functionality

---

## 📫 Contact & Support

### **Primary Contact**
- **Email**: Sylvesterobese6665@gmail.com
- **Phone**: +233(0) 540456262
- **GitHub**: [@Bempong-Sylvester-Obese](https://github.com/Bempong-Sylvester-Obese)

### **Support Channels**
- **Issues**: [GitHub Issues](https://github.com/Bempong-Sylvester-Obese/land-viability-checker/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Bempong-Sylvester-Obese/land-viability-checker/discussions)
- **Documentation**: [Project Wiki](https://github.com/Bempong-Sylvester-Obese/land-viability-checker/wiki)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Data Sources**: West African agricultural databases and research institutions
- **Open Source Libraries**: Scikit-learn, Pandas, NumPy, Matplotlib communities
- **Research Community**: Agricultural scientists and agronomists worldwide
- **Contributors**: All developers who have contributed to this project

---

💡 **"Empowering farmers with AI-driven insights for smarter agriculture."** 🌱

*Built with ❤️ for sustainable agriculture and food security.*


