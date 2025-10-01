import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SoilParameters:
    ph: float
    organic_matter: float  # %
    nitrogen: float        # ppm
    phosphorus: float      # ppm
    potassium: float       # ppm
    calcium: float         # ppm
    magnesium: float       # ppm
    sulfur: float          # ppm
    iron: float           # ppm
    manganese: float      # ppm
    zinc: float           # ppm
    copper: float         # ppm
    boron: float          # ppm
    clay_content: float   # %
    sand_content: float   # %
    silt_content: float   # %
    bulk_density: float   # g/cm³
    water_holding_capacity: float  # %
    cation_exchange_capacity: float  # meq/100g


class SoilQualityAnalyzer:
    def __init__(self):
        self.optimal_ranges = self._define_optimal_ranges()
        self.crop_requirements = self._define_crop_requirements()
        self.soil_types = self._define_soil_types()
    
    def _define_optimal_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'ph': (6.0, 7.5),  # Slightly acidic to neutral
            'organic_matter': (2.0, 5.0),  # %
            'nitrogen': (20, 40),  # ppm
            'phosphorus': (15, 30),  # ppm
            'potassium': (100, 200),  # ppm
            'calcium': (1000, 3000),  # ppm
            'magnesium': (100, 300),  # ppm
            'sulfur': (10, 20),  # ppm
            'iron': (10, 50),  # ppm
            'manganese': (5, 25),  # ppm
            'zinc': (1, 5),  # ppm
            'copper': (0.5, 2.0),  # ppm
            'boron': (0.5, 2.0),  # ppm
            'clay_content': (20, 40),  # %
            'sand_content': (40, 70),  # %
            'silt_content': (20, 40),  # %
            'bulk_density': (1.0, 1.4),  # g/cm³
            'water_holding_capacity': (15, 25),  # %
            'cation_exchange_capacity': (10, 25)  # meq/100g
        }
    
    def _define_crop_requirements(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        return {
            'maize': {
                'ph': (6.0, 7.5),
                'organic_matter': (2.0, 4.0),
                'nitrogen': (25, 50),
                'phosphorus': (15, 30),
                'potassium': (120, 200)
            },
            'rice': {
                'ph': (5.5, 7.0),
                'organic_matter': (2.0, 5.0),
                'nitrogen': (20, 40),
                'phosphorus': (10, 25),
                'potassium': (80, 150)
            },
            'wheat': {
                'ph': (6.0, 7.5),
                'organic_matter': (2.0, 4.0),
                'nitrogen': (25, 45),
                'phosphorus': (15, 30),
                'potassium': (100, 180)
            },
            'sorghum': {
                'ph': (6.0, 7.5),
                'organic_matter': (1.5, 3.5),
                'nitrogen': (20, 40),
                'phosphorus': (12, 25),
                'potassium': (100, 200)
            },
            'cassava': {
                'ph': (5.5, 7.0),
                'organic_matter': (2.0, 4.0),
                'nitrogen': (15, 35),
                'phosphorus': (10, 20),
                'potassium': (80, 150)
            },
            'yam': {
                'ph': (5.5, 7.0),
                'organic_matter': (2.5, 5.0),
                'nitrogen': (20, 40),
                'phosphorus': (12, 25),
                'potassium': (100, 200)
            }
        }
    
    def _define_soil_types(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        return {
            'clay': {'clay_content': (40, 100), 'sand_content': (0, 45), 'silt_content': (0, 40)},
            'clay_loam': {'clay_content': (27, 40), 'sand_content': (20, 45), 'silt_content': (15, 52)},
            'silty_clay': {'clay_content': (40, 60), 'sand_content': (0, 20), 'silt_content': (40, 60)},
            'silty_clay_loam': {'clay_content': (27, 40), 'sand_content': (0, 20), 'silt_content': (40, 73)},
            'sandy_clay': {'clay_content': (35, 55), 'sand_content': (45, 65), 'silt_content': (0, 20)},
            'sandy_clay_loam': {'clay_content': (20, 35), 'sand_content': (45, 80), 'silt_content': (0, 28)},
            'sandy_loam': {'clay_content': (7, 20), 'sand_content': (43, 85), 'silt_content': (0, 50)},
            'loam': {'clay_content': (7, 27), 'sand_content': (23, 52), 'silt_content': (28, 50)},
            'silt_loam': {'clay_content': (0, 27), 'sand_content': (0, 50), 'silt_content': (50, 88)},
            'silt': {'clay_content': (0, 12), 'sand_content': (0, 20), 'silt_content': (80, 100)},
            'loamy_sand': {'clay_content': (0, 15), 'sand_content': (70, 91), 'silt_content': (0, 30)},
            'sand': {'clay_content': (0, 10), 'sand_content': (85, 100), 'silt_content': (0, 15)}
        }
    
    def analyze_soil_parameters(self, soil: SoilParameters) -> Dict[str, Dict[str, Any]]:

        analysis = {}
        
        for param, value in soil.__dict__.items():
            if param in self.optimal_ranges:
                optimal_min, optimal_max = self.optimal_ranges[param]
                
                if optimal_min <= value <= optimal_max:
                    status = "Optimal"
                    score = 100
                elif value < optimal_min:
                    status = "Low"
                    score = max(0, (value / optimal_min) * 100)
                else:
                    status = "High"
                    score = max(0, 100 - ((value - optimal_max) / optimal_max) * 100)
                
                analysis[param] = {
                    'value': value,
                    'optimal_range': (optimal_min, optimal_max),
                    'status': status,
                    'score': round(score, 2),
                    'recommendation': self._get_parameter_recommendation(param, value, optimal_min, optimal_max)
                }
        
        return analysis
    
    def _get_parameter_recommendation(self, param: str, value: float, 
                                    optimal_min: float, optimal_max: float) -> str:

        recommendations = {
            'ph': {
                'low': "Apply lime to raise pH. Recommended: 2-4 tons/acre of agricultural lime.",
                'high': "Apply sulfur or acidifying fertilizers to lower pH."
            },
            'organic_matter': {
                'low': "Add compost, manure, or cover crops. Target: 2-3% organic matter.",
                'high': "Current levels are excellent. Maintain with regular organic additions."
            },
            'nitrogen': {
                'low': "Apply nitrogen fertilizer. Consider split applications during growing season.",
                'high': "Reduce nitrogen applications. Current levels may cause excessive growth."
            },
            'phosphorus': {
                'low': "Apply phosphorus fertilizer or rock phosphate. Incorporate into soil.",
                'high': "Phosphorus levels are adequate. Avoid additional P applications."
            },
            'potassium': {
                'low': "Apply potassium fertilizer (K₂O). Consider potash or wood ash.",
                'high': "Potassium levels are sufficient. Monitor for luxury consumption."
            }
        }
        
        if param in recommendations:
            if value < optimal_min:
                return recommendations[param]['low']
            elif value > optimal_max:
                return recommendations[param]['high']
        
        return "Levels are within optimal range. Continue current management practices."
    
    def calculate_soil_fertility_score(self, soil: SoilParameters) -> Dict[str, Any]:
        """
        Calculate overall soil fertility score.
        
        Args:
            soil: SoilParameters object
            
        Returns:
            Dictionary with fertility score and analysis
        """
        analysis = self.analyze_soil_parameters(soil)
        
        # Weight different parameters
        weights = {
            'ph': 0.15,
            'organic_matter': 0.20,
            'nitrogen': 0.15,
            'phosphorus': 0.15,
            'potassium': 0.10,
            'calcium': 0.05,
            'magnesium': 0.05,
            'clay_content': 0.05,
            'sand_content': 0.05,
            'silt_content': 0.05
        }
        
        weighted_score = 0
        total_weight = 0
        
        for param, data in analysis.items():
            if param in weights:
                weighted_score += data['score'] * weights[param]
                total_weight += weights[param]
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine fertility level
        if overall_score >= 80:
            fertility_level = "Excellent"
        elif overall_score >= 60:
            fertility_level = "Good"
        elif overall_score >= 40:
            fertility_level = "Fair"
        else:
            fertility_level = "Poor"
        
        return {
            'overall_score': round(overall_score, 2),
            'fertility_level': fertility_level,
            'parameter_analysis': analysis,
            'recommendations': self._get_overall_recommendations(analysis, overall_score)
        }
    
    def _get_overall_recommendations(self, analysis: Dict, overall_score: float) -> List[str]:
        recommendations = []
        
        if overall_score < 40:
            recommendations.append("CRITICAL: Soil fertility is very low. Immediate soil improvement needed.")
            recommendations.append("Priority actions: Add organic matter, correct pH, apply balanced fertilizers.")
        elif overall_score < 60:
            recommendations.append("MODERATE: Soil fertility needs improvement for optimal crop production.")
            recommendations.append("Focus on: Organic matter addition, nutrient balancing, pH correction.")
        elif overall_score < 80:
            recommendations.append("GOOD: Soil fertility is acceptable but can be improved.")
            recommendations.append("Maintain current practices and address specific nutrient deficiencies.")
        else:
            recommendations.append("EXCELLENT: Soil fertility is optimal for crop production.")
            recommendations.append("Continue current management practices to maintain soil health.")
        
        # Add specific recommendations for low-scoring parameters
        low_params = [param for param, data in analysis.items() if data['score'] < 60]
        if low_params:
            recommendations.append(f"Priority nutrients to address: {', '.join(low_params)}")
        
        return recommendations
    
    def classify_soil_type(self, soil: SoilParameters) -> str:
        """
        Classify soil type based on texture analysis.
        
        Args:
            soil: SoilParameters object
            
        Returns:
            Soil type classification
        """
        clay = soil.clay_content
        sand = soil.sand_content
        silt = soil.silt_content
        
        # Normalize percentages (should sum to 100)
        total = clay + sand + silt
        if total > 0:
            clay = (clay / total) * 100
            sand = (sand / total) * 100
            silt = (silt / total) * 100
        
        # Check against soil type definitions
        for soil_type, ranges in self.soil_types.items():
            clay_range = ranges['clay_content']
            sand_range = ranges['sand_content']
            silt_range = ranges['silt_content']
            
            if (clay_range[0] <= clay <= clay_range[1] and
                sand_range[0] <= sand <= sand_range[1] and
                silt_range[0] <= silt <= silt_range[1]):
                return soil_type
        
        return "Unknown"
    
    def assess_crop_suitability(self, soil: SoilParameters, crop: str) -> Dict[str, Any]:
        """
        Assess soil suitability for specific crops.
        
        Args:
            soil: SoilParameters object
            crop: Crop name (maize, rice, wheat, etc.)
            
        Returns:
            Dictionary with crop suitability assessment
        """
        if crop not in self.crop_requirements:
            return {"error": f"Crop '{crop}' not in database"}
        
        requirements = self.crop_requirements[crop]
        soil_analysis = self.analyze_soil_parameters(soil)
        
        suitability_scores = []
        limiting_factors = []
        
        for param, (min_req, max_req) in requirements.items():
            if param in soil_analysis:
                soil_value = soil_analysis[param]['value']
                
                if min_req <= soil_value <= max_req:
                    score = 100
                elif soil_value < min_req:
                    score = (soil_value / min_req) * 100
                    limiting_factors.append(f"{param} too low ({soil_value:.1f} < {min_req:.1f})")
                else:
                    score = max(0, 100 - ((soil_value - max_req) / max_req) * 100)
                    limiting_factors.append(f"{param} too high ({soil_value:.1f} > {max_req:.1f})")
                
                suitability_scores.append(score)
        
        overall_suitability = np.mean(suitability_scores) if suitability_scores else 0
        
        # Determine suitability level
        if overall_suitability >= 80:
            suitability_level = "Highly Suitable"
        elif overall_suitability >= 60:
            suitability_level = "Moderately Suitable"
        elif overall_suitability >= 40:
            suitability_level = "Marginally Suitable"
        else:
            suitability_level = "Unsuitable"
        
        return {
            'crop': crop,
            'suitability_score': round(overall_suitability, 2),
            'suitability_level': suitability_level,
            'limiting_factors': limiting_factors,
            'recommendations': self._get_crop_specific_recommendations(crop, limiting_factors)
        }
    
    def _get_crop_specific_recommendations(self, crop: str, limiting_factors: List[str]) -> List[str]:
        recommendations = []
        
        crop_recs = {
            'maize': "For maize: Ensure adequate N and P, maintain good drainage, use crop rotation.",
            'rice': "For rice: Maintain slightly acidic pH, ensure good water management, add organic matter.",
            'wheat': "For wheat: Focus on N and P nutrition, maintain neutral pH, ensure good soil structure.",
            'sorghum': "For sorghum: Drought-tolerant crop, ensure good drainage, moderate fertility needs.",
            'cassava': "For cassava: Tolerant of low fertility, ensure good drainage, avoid waterlogging.",
            'yam': "For yam: Requires well-drained soil, high organic matter, good soil structure."
        }
        
        if crop in crop_recs:
            recommendations.append(crop_recs[crop])
        
        if limiting_factors:
            recommendations.append("Address limiting factors: " + ", ".join(limiting_factors[:3]))
        
        return recommendations
    
    def create_soil_report(self, soil: SoilParameters, crops: Optional[List[str]] = None) -> str:
        """
        Create a comprehensive soil analysis report.
        
        Args:
            soil: SoilParameters object
            crops: List of crops to assess suitability for
            
        Returns:
            Formatted soil analysis report
        """
        if crops is None:
            crops = ['maize', 'rice', 'wheat', 'sorghum']
        
        # Get soil type
        soil_type = self.classify_soil_type(soil)
        
        # Get fertility analysis
        fertility_analysis = self.calculate_soil_fertility_score(soil)
        
        # Get crop suitability assessments
        crop_assessments = {}
        for crop in crops:
            crop_assessments[crop] = self.assess_crop_suitability(soil, crop)
        
        # Create report
        report = f"""
SOIL ANALYSIS REPORT
{'='*50}

SOIL CHARACTERISTICS:
   Soil Type: {soil_type}
   pH: {soil.ph:.2f}
   Organic Matter: {soil.organic_matter:.1f}%
   Clay: {soil.clay_content:.1f}%, Sand: {soil.sand_content:.1f}%, Silt: {soil.silt_content:.1f}%

FERTILITY ASSESSMENT:
   Overall Score: {fertility_analysis['overall_score']}/100
   Fertility Level: {fertility_analysis['fertility_level']}

RECOMMENDATIONS:
"""
        
        for rec in fertility_analysis['recommendations']:
            report += f"   {rec}\n"
        
        report += f"\nCROP SUITABILITY ANALYSIS:\n"
        for crop, assessment in crop_assessments.items():
            report += f"\n   {crop.upper()}: {assessment['suitability_level']} ({assessment['suitability_score']:.1f}/100)\n"
            if assessment['limiting_factors']:
                report += f"      Limiting factors: {', '.join(assessment['limiting_factors'][:2])}\n"
        
        report += f"\n{'='*50}\n"
        report += f"Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return report
    
    def plot_soil_analysis(self, soil: SoilParameters, save_path: Optional[str] = None):
        """
        Create visualizations for soil analysis.
        
        Args:
            soil: SoilParameters object
            save_path: Path to save the plot
        """
        analysis = self.analyze_soil_parameters(soil)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Parameter scores
        params = list(analysis.keys())[:10]  # Top 10 parameters
        scores = [analysis[param]['score'] for param in params]
        colors = ['green' if score >= 80 else 'orange' if score >= 60 else 'red' for score in scores]
        
        axes[0, 0].barh(params, scores, color=colors, alpha=0.7)
        axes[0, 0].set_title('Soil Parameter Scores')
        axes[0, 0].set_xlabel('Score (0-100)')
        axes[0, 0].axvline(x=80, color='green', linestyle='--', alpha=0.5, label='Optimal')
        axes[0, 0].axvline(x=60, color='orange', linestyle='--', alpha=0.5, label='Acceptable')
        axes[0, 0].legend()
        
        # Soil texture triangle (simplified)
        clay = soil.clay_content
        sand = soil.sand_content
        silt = soil.silt_content
        
        axes[0, 1].scatter(sand, clay, s=200, c='red', alpha=0.7)
        axes[0, 1].set_xlabel('Sand Content (%)')
        axes[0, 1].set_ylabel('Clay Content (%)')
        axes[0, 1].set_title('Soil Texture')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Nutrient levels
        nutrients = ['nitrogen', 'phosphorus', 'potassium', 'calcium', 'magnesium']
        nutrient_values = [getattr(soil, nutrient) for nutrient in nutrients if hasattr(soil, nutrient)]
        nutrient_labels = [nutrient.title() for nutrient in nutrients if hasattr(soil, nutrient)]
        
        if nutrient_values:
            axes[1, 0].bar(nutrient_labels, nutrient_values, color='skyblue', alpha=0.7)
            axes[1, 0].set_title('Major Nutrients (ppm)')
            axes[1, 0].set_ylabel('Concentration (ppm)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Overall fertility score
        fertility = self.calculate_soil_fertility_score(soil)
        score = fertility['overall_score']
        level = fertility['fertility_level']
        
        colors_fertility = {'Excellent': 'green', 'Good': 'lightgreen', 'Fair': 'orange', 'Poor': 'red'}
        axes[1, 1].pie([score, 100-score], labels=[f'{level}\n{score}%', ''], 
                      colors=[colors_fertility.get(level, 'gray'), 'lightgray'],
                      autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Overall Fertility Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Soil analysis plot saved to {save_path}")
        
        plt.show()


def create_sample_soil_data() -> SoilParameters:
    """Create sample soil data for testing."""
    return SoilParameters(
        ph=6.2,
        organic_matter=2.5,
        nitrogen=25.0,
        phosphorus=18.0,
        potassium=150.0,
        calcium=1200.0,
        magnesium=150.0,
        sulfur=12.0,
        iron=25.0,
        manganese=12.0,
        zinc=2.0,
        copper=1.0,
        boron=0.8,
        clay_content=25.0,
        sand_content=55.0,
        silt_content=20.0,
        bulk_density=1.2,
        water_holding_capacity=18.0,
        cation_exchange_capacity=15.0
    )


if __name__ == "__main__":
    # Create sample soil data
    soil = create_sample_soil_data()
    
    # Initialize analyzer
    analyzer = SoilQualityAnalyzer()
    
    # Perform analysis
    print("Soil Quality Analysis Demo")
    print("="*50)
    
    # Generate report
    report = analyzer.create_soil_report(soil, ['maize', 'rice', 'wheat', 'sorghum'])
    print(report)
    
    # Create visualizations
    analyzer.plot_soil_analysis(soil, "Data/soil_analysis.png")
    
    print("Soil analysis completed successfully!")
