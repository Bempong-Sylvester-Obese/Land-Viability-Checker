"""
Land Viability Assessment Module
This module integrates soil, climate, and crop yield analysis for comprehensive land assessment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import os
from datetime import datetime

# Import our custom modules
from .soil_analysis import SoilParameters, SoilQualityAnalyzer
from .climate_analysis import ClimateParameters, ClimateAnalyzer
from .crop_yield_models import CropYieldPredictor


@dataclass
class LandParameters:
    """Comprehensive land parameters combining soil, climate, and location data."""
    # Location (required fields first)
    latitude: float
    longitude: float
    elevation: float  # meters
    
    # Soil parameters (required fields)
    soil_ph: float
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
    
    # Climate parameters (required fields)
    temperature_avg: float  # °C
    temperature_min: float  # °C
    temperature_max: float  # °C
    rainfall_annual: float  # mm
    rainfall_seasonal: float  # mm
    humidity_avg: float  # %
    sunshine_hours: float  # hours/day
    wind_speed: float  # m/s
    
    # Optional fields (with defaults)
    location_name: str = "Unknown"
    slope: float = 0.0  # degrees
    drainage: str = "Good"  # Good, Fair, Poor
    irrigation_available: bool = False
    market_access: str = "Good"  # Good, Fair, Poor


class LandViabilityAssessor:
    """
    Comprehensive land viability assessment system that integrates:
    - Soil quality analysis
    - Climate suitability assessment
    - Crop yield predictions
    - Economic viability factors
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the Land Viability Assessor.
        
        Args:
            model_dir: Directory containing trained ML models
        """
        self.model_dir = model_dir
        self.soil_analyzer = SoilQualityAnalyzer()
        self.climate_analyzer = ClimateAnalyzer()
        self.crop_predictor = None
        
        # Load crop yield models if available
        self._load_models()
        
        # Economic factors
        self.crop_prices = self._define_crop_prices()
        self.crop_costs = self._define_crop_costs()
        self.crop_requirements = self._define_crop_requirements()
    
    def _load_models(self):
        """Load trained crop yield prediction models."""
        try:
            self.crop_predictor = CropYieldPredictor(self.model_dir)
            if os.path.exists(os.path.join(self.model_dir, "crop_yield_scores.json")):
                self.crop_predictor.load_models()
                print("Crop yield models loaded successfully")
            else:
                print("No trained crop yield models found. Train models first.")
        except Exception as e:
            print(f"Error loading crop yield models: {e}")
            self.crop_predictor = None
    
    def _define_crop_prices(self) -> Dict[str, float]:
        """Define current crop prices (USD per ton)."""
        return {
            'maize': 280.0,
            'rice': 450.0,
            'wheat': 320.0,
            'sorghum': 250.0,
            'cassava': 180.0,
            'yam': 400.0,
            'beans': 600.0,
            'groundnuts': 800.0
        }
    
    def _define_crop_costs(self) -> Dict[str, float]:
        """Define production costs (USD per hectare)."""
        return {
            'maize': 800.0,
            'rice': 1200.0,
            'wheat': 900.0,
            'sorghum': 600.0,
            'cassava': 500.0,
            'yam': 700.0,
            'beans': 600.0,
            'groundnuts': 800.0
        }
    
    def _define_crop_requirements(self) -> Dict[str, Dict[str, float]]:
        """Define crop-specific requirements."""
        return {
            'maize': {
                'min_ph': 6.0,
                'min_rainfall': 500,
                'growing_season': 120,
                'yield_target': 3.0  # tons/hectare
            },
            'rice': {
                'min_ph': 5.5,
                'min_rainfall': 1000,
                'growing_season': 150,
                'yield_target': 4.0
            },
            'wheat': {
                'min_ph': 6.0,
                'min_rainfall': 400,
                'growing_season': 180,
                'yield_target': 2.5
            },
            'sorghum': {
                'min_ph': 6.0,
                'min_rainfall': 300,
                'growing_season': 120,
                'yield_target': 2.0
            },
            'cassava': {
                'min_ph': 5.5,
                'min_rainfall': 800,
                'growing_season': 300,
                'yield_target': 15.0
            },
            'yam': {
                'min_ph': 5.5,
                'min_rainfall': 1000,
                'growing_season': 240,
                'yield_target': 10.0
            }
        }
    
    def convert_to_soil_parameters(self, land: LandParameters) -> SoilParameters:
        """Convert LandParameters to SoilParameters."""
        return SoilParameters(
            ph=land.soil_ph,
            organic_matter=land.organic_matter,
            nitrogen=land.nitrogen,
            phosphorus=land.phosphorus,
            potassium=land.potassium,
            calcium=land.calcium,
            magnesium=land.magnesium,
            sulfur=land.sulfur,
            iron=land.iron,
            manganese=land.manganese,
            zinc=land.zinc,
            copper=land.copper,
            boron=land.boron,
            clay_content=land.clay_content,
            sand_content=land.sand_content,
            silt_content=land.silt_content,
            bulk_density=land.bulk_density,
            water_holding_capacity=land.water_holding_capacity,
            cation_exchange_capacity=land.cation_exchange_capacity
        )
    
    def convert_to_climate_parameters(self, land: LandParameters) -> ClimateParameters:
        """Convert LandParameters to ClimateParameters."""
        return ClimateParameters(
            temperature_avg=land.temperature_avg,
            temperature_min=land.temperature_min,
            temperature_max=land.temperature_max,
            rainfall_annual=land.rainfall_annual,
            rainfall_seasonal=land.rainfall_seasonal,
            humidity_avg=land.humidity_avg,
            sunshine_hours=land.sunshine_hours,
            wind_speed=land.wind_speed,
            elevation=land.elevation,
            latitude=land.latitude,
            longitude=land.longitude
        )
    
    def assess_land_viability(self, land: LandParameters, 
                            crops: List[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive land viability assessment.
        
        Args:
            land: LandParameters object
            crops: List of crops to assess (default: all available crops)
            
        Returns:
            Dictionary with comprehensive viability assessment
        """
        if crops is None:
            crops = list(self.crop_requirements.keys())
        
        print(f"Assessing land viability for {land.location_name}...")
        
        # Convert to specialized parameter objects
        soil_params = self.convert_to_soil_parameters(land)
        climate_params = self.convert_to_climate_parameters(land)
        
        # Perform individual analyses
        soil_analysis = self.soil_analyzer.calculate_soil_fertility_score(soil_params)
        climate_analysis = self.climate_analyzer.calculate_climate_suitability_score(climate_params)
        
        # Assess each crop
        crop_assessments = {}
        for crop in crops:
            crop_assessments[crop] = self._assess_crop_viability(
                land, soil_params, climate_params, crop
            )
        
        # Calculate overall land viability
        overall_viability = self._calculate_overall_viability(
            soil_analysis, climate_analysis, crop_assessments, land
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            soil_analysis, climate_analysis, crop_assessments, land
        )
        
        return {
            'location': land.location_name,
            'coordinates': (land.latitude, land.longitude),
            'assessment_date': datetime.now().isoformat(),
            'soil_analysis': soil_analysis,
            'climate_analysis': climate_analysis,
            'crop_assessments': crop_assessments,
            'overall_viability': overall_viability,
            'recommendations': recommendations,
            'economic_analysis': self._perform_economic_analysis(crop_assessments)
        }
    
    def _assess_crop_viability(self, land: LandParameters, soil_params: SoilParameters,
                             climate_params: ClimateParameters, crop: str) -> Dict[str, Any]:
        """Assess viability for a specific crop."""
        # Soil suitability
        soil_suitability = self.soil_analyzer.assess_crop_suitability(soil_params, crop)
        
        # Climate suitability
        climate_suitability = self.climate_analyzer.assess_crop_climate_suitability(climate_params, crop)
        
        # Physical constraints
        physical_constraints = self._assess_physical_constraints(land, crop)
        
        # Yield prediction (if models available)
        predicted_yield = None
        if self.crop_predictor and crop in ['maize', 'rice', 'wheat', 'sorghum']:
            predicted_yield = self._predict_crop_yield(land, crop)
        
        # Calculate overall crop viability score
        scores = [
            soil_suitability['suitability_score'],
            climate_suitability['suitability_score'],
            physical_constraints['suitability_score']
        ]
        
        overall_score = np.mean(scores)
        
        if overall_score >= 80:
            viability_level = "Highly Viable"
        elif overall_score >= 60:
            viability_level = "Moderately Viable"
        elif overall_score >= 40:
            viability_level = "Marginally Viable"
        else:
            viability_level = "Not Viable"
        
        return {
            'crop': crop,
            'overall_score': round(overall_score, 2),
            'viability_level': viability_level,
            'soil_suitability': soil_suitability,
            'climate_suitability': climate_suitability,
            'physical_constraints': physical_constraints,
            'predicted_yield': predicted_yield,
            'limiting_factors': self._identify_limiting_factors(
                soil_suitability, climate_suitability, physical_constraints
            )
        }
    
    def _assess_physical_constraints(self, land: LandParameters, crop: str) -> Dict[str, Any]:
        """Assess physical constraints for crop production."""
        constraints = []
        scores = []
        
        # Slope constraint
        if land.slope > 15:
            constraints.append(f"High slope ({land.slope:.1f}°) - erosion risk")
            scores.append(40)
        elif land.slope > 8:
            constraints.append(f"Moderate slope ({land.slope:.1f}°) - requires terracing")
            scores.append(70)
        else:
            scores.append(100)
        
        # Drainage constraint
        if land.drainage == "Poor":
            constraints.append("Poor drainage - waterlogging risk")
            scores.append(30)
        elif land.drainage == "Fair":
            constraints.append("Fair drainage - may need improvement")
            scores.append(70)
        else:
            scores.append(100)
        
        # Irrigation constraint
        crop_req = self.crop_requirements.get(crop, {})
        min_rainfall = crop_req.get('min_rainfall', 0)
        
        if land.rainfall_annual < min_rainfall and not land.irrigation_available:
            constraints.append(f"Insufficient rainfall ({land.rainfall_annual:.0f}mm < {min_rainfall}mm)")
            scores.append(20)
        else:
            scores.append(100)
        
        overall_score = np.mean(scores) if scores else 100
        
        return {
            'suitability_score': round(overall_score, 2),
            'constraints': constraints,
            'slope_score': scores[0] if len(scores) > 0 else 100,
            'drainage_score': scores[1] if len(scores) > 1 else 100,
            'water_score': scores[2] if len(scores) > 2 else 100
        }
    
    def _predict_crop_yield(self, land: LandParameters, crop: str) -> Dict[str, Any]:
        """Predict crop yield using trained models."""
        if not self.crop_predictor:
            return None
        
        try:
            # Create feature vector for prediction
            features = pd.DataFrame({
                'Year_Numeric': [2024],
                'Country_Encoded': [0],  # Default to first country
                'Decade': [2020]
            })
            
            # Add soil and climate features
            features['Soil_Ph'] = land.soil_ph
            features['Organic_Matter'] = land.organic_matter
            features['Nitrogen'] = land.nitrogen
            features['Phosphorus'] = land.phosphorus
            features['Potassium'] = land.potassium
            features['Temperature_Avg'] = land.temperature_avg
            features['Rainfall_Annual'] = land.rainfall_annual
            
            # Fill missing features with defaults
            for feature in self.crop_predictor.models[self.crop_predictor.best_model].feature_names_in_:
                if feature not in features.columns:
                    features[feature] = 0
            
            # Make prediction
            prediction = self.crop_predictor.predict(features, self.crop_predictor.best_model)
            
            return {
                'predicted_yield': round(prediction[0], 2),
                'model_used': self.crop_predictor.best_model,
                'confidence': 'Medium'  # Could be enhanced with actual confidence intervals
            }
            
        except Exception as e:
            print(f"Error predicting yield for {crop}: {e}")
            return None
    
    def _identify_limiting_factors(self, soil_suitability: Dict, climate_suitability: Dict,
                                 physical_constraints: Dict) -> List[str]:
        """Identify the main limiting factors for crop production."""
        limiting_factors = []
        
        # Soil limiting factors
        if soil_suitability['suitability_score'] < 60:
            limiting_factors.extend(soil_suitability['limiting_factors'][:2])
        
        # Climate limiting factors
        if climate_suitability['suitability_score'] < 60:
            limiting_factors.extend(climate_suitability['limiting_factors'][:2])
        
        # Physical limiting factors
        if physical_constraints['suitability_score'] < 60:
            limiting_factors.extend(physical_constraints['constraints'][:2])
        
        return limiting_factors[:5]  # Return top 5 limiting factors
    
    def _calculate_overall_viability(self, soil_analysis: Dict, climate_analysis: Dict,
                                   crop_assessments: Dict, land: LandParameters) -> Dict[str, Any]:
        """Calculate overall land viability score."""
        # Weight different components
        soil_score = soil_analysis['overall_score']
        climate_score = climate_analysis['overall_score']
        
        # Average crop viability scores
        crop_scores = [assessment['overall_score'] for assessment in crop_assessments.values()]
        avg_crop_score = np.mean(crop_scores) if crop_scores else 0
        
        # Physical factors
        physical_score = 100
        if land.slope > 15:
            physical_score -= 20
        if land.drainage == "Poor":
            physical_score -= 30
        if not land.irrigation_available and land.rainfall_annual < 600:
            physical_score -= 25
        
        # Market access factor
        market_score = 100
        if land.market_access == "Poor":
            market_score = 60
        elif land.market_access == "Fair":
            market_score = 80
        
        # Calculate weighted overall score
        weights = {
            'soil': 0.25,
            'climate': 0.25,
            'crops': 0.25,
            'physical': 0.15,
            'market': 0.10
        }
        
        overall_score = (
            soil_score * weights['soil'] +
            climate_score * weights['climate'] +
            avg_crop_score * weights['crops'] +
            physical_score * weights['physical'] +
            market_score * weights['market']
        )
        
        if overall_score >= 80:
            viability_level = "Excellent"
        elif overall_score >= 60:
            viability_level = "Good"
        elif overall_score >= 40:
            viability_level = "Fair"
        else:
            viability_level = "Poor"
        
        return {
            'overall_score': round(overall_score, 2),
            'viability_level': viability_level,
            'component_scores': {
                'soil': soil_score,
                'climate': climate_score,
                'crops': round(avg_crop_score, 2),
                'physical': physical_score,
                'market': market_score
            }
        }
    
    def _perform_economic_analysis(self, crop_assessments: Dict) -> Dict[str, Any]:
        """Perform economic viability analysis."""
        economic_results = {}
        
        for crop, assessment in crop_assessments.items():
            if crop in self.crop_prices and crop in self.crop_costs:
                price = self.crop_prices[crop]
                cost = self.crop_costs[crop]
                
                # Estimate yield based on viability score
                estimated_yield = assessment['predicted_yield']
                if estimated_yield is None:
                    # Use target yield scaled by viability score
                    target_yield = self.crop_requirements[crop]['yield_target']
                    viability_factor = assessment['overall_score'] / 100
                    estimated_yield = target_yield * viability_factor
                
                # Calculate revenue and profit
                revenue = estimated_yield['predicted_yield'] * price if isinstance(estimated_yield, dict) else estimated_yield * price
                profit = revenue - cost
                profit_margin = (profit / revenue * 100) if revenue > 0 else 0
                
                economic_results[crop] = {
                    'estimated_yield': round(estimated_yield['predicted_yield'] if isinstance(estimated_yield, dict) else estimated_yield, 2),
                    'price_per_ton': price,
                    'cost_per_hectare': cost,
                    'estimated_revenue': round(revenue, 2),
                    'estimated_profit': round(profit, 2),
                    'profit_margin': round(profit_margin, 2),
                    'economic_viability': 'High' if profit_margin > 20 else 'Medium' if profit_margin > 10 else 'Low'
                }
        
        return economic_results
    
    def _generate_recommendations(self, soil_analysis: Dict, climate_analysis: Dict,
                                crop_assessments: Dict, land: LandParameters) -> List[str]:
        """Generate comprehensive recommendations."""
        recommendations = []
        
        # Overall recommendations
        overall_score = (
            soil_analysis['overall_score'] + 
            climate_analysis['overall_score']
        ) / 2
        
        if overall_score < 40:
            recommendations.append("CRITICAL: This land requires significant improvement before agricultural use.")
            recommendations.append("Priority: Soil improvement, water management, and infrastructure development.")
        elif overall_score < 60:
            recommendations.append("MODERATE: Land has potential but requires careful management and improvements.")
            recommendations.append("Focus: Soil fertility, crop selection, and sustainable practices.")
        elif overall_score < 80:
            recommendations.append("GOOD: Land is suitable for agriculture with proper management.")
            recommendations.append("Maintain: Current practices and address specific limitations.")
        else:
            recommendations.append("EXCELLENT: Land is highly suitable for agricultural production.")
            recommendations.append("Continue: Current practices and optimize for maximum productivity.")
        
        # Specific recommendations
        best_crops = sorted(crop_assessments.items(), key=lambda x: x[1]['overall_score'], reverse=True)[:3]
        recommendations.append(f"Recommended crops (in order): {', '.join([crop for crop, _ in best_crops])}")
        
        # Soil recommendations
        if soil_analysis['overall_score'] < 60:
            recommendations.append("Soil improvement needed: Add organic matter, correct pH, balance nutrients")
        
        # Climate recommendations
        if climate_analysis['overall_score'] < 60:
            recommendations.append("Climate management: Consider irrigation, crop timing, or protected cultivation")
        
        # Physical recommendations
        if land.slope > 8:
            recommendations.append("Erosion control: Implement terracing, contour farming, or cover crops")
        
        if land.drainage == "Poor":
            recommendations.append("Drainage improvement: Install drainage systems or raised beds")
        
        return recommendations
    
    def create_comprehensive_report(self, land: LandParameters, 
                                  crops: List[str] = None) -> str:
        """Create a comprehensive land viability report."""
        assessment = self.assess_land_viability(land, crops)
        
        report = f"""
LAND VIABILITY ASSESSMENT REPORT
{'='*60}

PROPERTY INFORMATION:
   Location: {land.location_name}
   Coordinates: {land.latitude:.4f}, {land.longitude:.4f}
   Elevation: {land.elevation:.0f} meters
   Assessment Date: {assessment['assessment_date']}

OVERALL VIABILITY ASSESSMENT:
   Overall Score: {assessment['overall_viability']['overall_score']}/100
   Viability Level: {assessment['overall_viability']['viability_level']}

COMPONENT SCORES:
   Soil Quality: {assessment['overall_viability']['component_scores']['soil']}/100
   Climate Suitability: {assessment['overall_viability']['component_scores']['climate']}/100
   Crop Potential: {assessment['overall_viability']['component_scores']['crops']}/100
   Physical Factors: {assessment['overall_viability']['component_scores']['physical']}/100
   Market Access: {assessment['overall_viability']['component_scores']['market']}/100

CROP VIABILITY ANALYSIS:
"""
        
        for crop, crop_data in assessment['crop_assessments'].items():
            report += f"\n   {crop.upper()}: {crop_data['viability_level']} ({crop_data['overall_score']}/100)\n"
            
            if crop_data['predicted_yield']:
                yield_info = crop_data['predicted_yield']
                if isinstance(yield_info, dict):
                    report += f"      Predicted Yield: {yield_info['predicted_yield']:.2f} tons/hectare\n"
            
            if crop_data['limiting_factors']:
                report += f"      Limiting Factors: {', '.join(crop_data['limiting_factors'][:3])}\n"
        
        report += f"\nECONOMIC ANALYSIS:\n"
        for crop, econ_data in assessment['economic_analysis'].items():
            report += f"\n   {crop.upper()}:\n"
            report += f"      Estimated Yield: {econ_data['estimated_yield']:.2f} tons/hectare\n"
            report += f"      Revenue: ${econ_data['estimated_revenue']:.2f}/hectare\n"
            report += f"      Profit: ${econ_data['estimated_profit']:.2f}/hectare\n"
            report += f"      Margin: {econ_data['profit_margin']:.1f}%\n"
        
        report += f"\nRECOMMENDATIONS:\n"
        for i, rec in enumerate(assessment['recommendations'], 1):
            report += f"   {i}. {rec}\n"
        
        report += f"\n{'='*60}\n"
        report += f"Report generated by Land Viability Checker\n"
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return report
    
    def save_assessment(self, assessment: Dict[str, Any], filename: str):
        """Save assessment results to JSON file."""
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(assessment, f, indent=2, default=str)
        
        print(f"Assessment saved to {filename}")
    
    def plot_viability_dashboard(self, assessment: Dict[str, Any], save_path: str = None):
        """Create a comprehensive viability dashboard visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Overall viability score
        overall_score = assessment['overall_viability']['overall_score']
        colors = ['red' if overall_score < 40 else 'orange' if overall_score < 60 else 'lightgreen' if overall_score < 80 else 'green']
        
        axes[0, 0].pie([overall_score, 100-overall_score], 
                      labels=[f'{assessment["overall_viability"]["viability_level"]}\n{overall_score}%', ''], 
                      colors=[colors[0], 'lightgray'], autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Overall Viability Score')
        
        # Component scores
        components = assessment['overall_viability']['component_scores']
        comp_names = list(components.keys())
        comp_scores = list(components.values())
        
        bars = axes[0, 1].bar(comp_names, comp_scores, color=['skyblue', 'lightgreen', 'orange', 'pink', 'yellow'])
        axes[0, 1].set_title('Component Scores')
        axes[0, 1].set_ylabel('Score (0-100)')
        axes[0, 1].set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, score in zip(bars, comp_scores):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                           f'{score:.0f}', ha='center', va='bottom')
        
        # Crop viability scores
        crops = list(assessment['crop_assessments'].keys())
        crop_scores = [assessment['crop_assessments'][crop]['overall_score'] for crop in crops]
        
        bars = axes[0, 2].bar(crops, crop_scores, color='lightcoral', alpha=0.7)
        axes[0, 2].set_title('Crop Viability Scores')
        axes[0, 2].set_ylabel('Score (0-100)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].set_ylim(0, 100)
        
        # Economic analysis
        if assessment['economic_analysis']:
            crops_econ = list(assessment['economic_analysis'].keys())
            profits = [assessment['economic_analysis'][crop]['estimated_profit'] for crop in crops_econ]
            
            colors_econ = ['green' if p > 500 else 'orange' if p > 0 else 'red' for p in profits]
            bars = axes[1, 0].bar(crops_econ, profits, color=colors_econ, alpha=0.7)
            axes[1, 0].set_title('Estimated Profit per Hectare')
            axes[1, 0].set_ylabel('Profit ($)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Soil vs Climate suitability
        soil_score = assessment['soil_analysis']['overall_score']
        climate_score = assessment['climate_analysis']['overall_score']
        
        axes[1, 1].scatter([soil_score], [climate_score], s=200, c='blue', alpha=0.7)
        axes[1, 1].set_xlabel('Soil Quality Score')
        axes[1, 1].set_ylabel('Climate Suitability Score')
        axes[1, 1].set_title('Soil vs Climate Suitability')
        axes[1, 1].set_xlim(0, 100)
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add quadrant lines
        axes[1, 1].axhline(y=60, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].axvline(x=60, color='red', linestyle='--', alpha=0.5)
        
        # Limiting factors summary
        all_limiting_factors = []
        for crop_data in assessment['crop_assessments'].values():
            all_limiting_factors.extend(crop_data['limiting_factors'])
        
        # Count limiting factors
        factor_counts = {}
        for factor in all_limiting_factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        if factor_counts:
            top_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            factors, counts = zip(*top_factors)
            
            axes[1, 2].barh(factors, counts, color='salmon', alpha=0.7)
            axes[1, 2].set_title('Most Common Limiting Factors')
            axes[1, 2].set_xlabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Viability dashboard saved to {save_path}")
        
        plt.show()


def create_sample_land_data() -> LandParameters:
    """Create sample land data for testing."""
    return LandParameters(
        latitude=6.5,
        longitude=-1.6,
        elevation=150.0,
        location_name="Kumasi, Ghana",
        soil_ph=6.2,
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
        cation_exchange_capacity=15.0,
        temperature_avg=26.5,
        temperature_min=22.0,
        temperature_max=32.0,
        rainfall_annual=1200.0,
        rainfall_seasonal=600.0,
        humidity_avg=70.0,
        sunshine_hours=8.5,
        wind_speed=3.0,
        slope=5.0,
        drainage="Good",
        irrigation_available=True,
        market_access="Good"
    )


if __name__ == "__main__":
    # Create sample land data
    land = create_sample_land_data()
    
    # Initialize assessor
    assessor = LandViabilityAssessor()
    
    # Perform assessment
    print("Land Viability Assessment Demo")
    print("="*50)
    
    # Generate comprehensive report
    report = assessor.create_comprehensive_report(land, ['maize', 'rice', 'wheat', 'sorghum'])
    print(report)
    
    # Perform full assessment
    assessment = assessor.assess_land_viability(land, ['maize', 'rice', 'wheat', 'sorghum'])
    
    # Create visualizations
    assessor.plot_viability_dashboard(assessment, "Data/viability_dashboard.png")
    
    # Save assessment
    assessor.save_assessment(assessment, "Data/sample_assessment.json")
    
    print("Land viability assessment completed successfully!")
