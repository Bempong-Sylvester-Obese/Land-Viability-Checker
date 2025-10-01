import pandas as pd
import numpy as np
import requests
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ClimateParameters:
    temperature_avg: float  # °C
    temperature_min: float  # °C
    temperature_max: float  # °C
    rainfall_annual: float  # mm
    rainfall_seasonal: float  # mm (growing season)
    humidity_avg: float  # %
    sunshine_hours: float  # hours/day
    wind_speed: float  # m/s
    elevation: float  # meters above sea level
    latitude: float
    longitude: float


class ClimateAnalyzer:
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Climate Analyzer.
        
        Args:
            api_key: API key for weather services (optional)
        """
        self.api_key = api_key
        self.optimal_climate_ranges = self._define_optimal_climate_ranges()
        self.crop_climate_requirements = self._define_crop_climate_requirements()
        self.climate_zones = self._define_climate_zones()
    
    def _define_optimal_climate_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Define optimal climate ranges for general agriculture."""
        return {
            'temperature_avg': (20, 30),  # °C
            'temperature_min': (15, 25),  # °C
            'temperature_max': (25, 35),  # °C
            'rainfall_annual': (800, 1500),  # mm
            'rainfall_seasonal': (400, 800),  # mm
            'humidity_avg': (60, 80),  # %
            'sunshine_hours': (6, 10),  # hours/day
            'wind_speed': (1, 5),  # m/s
            'elevation': (0, 1500)  # meters
        }
    
    def _define_crop_climate_requirements(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Define crop-specific climate requirements."""
        return {
            'maize': {
                'temperature_avg': (18, 27),
                'rainfall_annual': (500, 1000),
                'rainfall_seasonal': (300, 600),
                'humidity_avg': (50, 70),
                'sunshine_hours': (8, 12),
                'growing_season': (120, 120)  # days
            },
            'rice': {
                'temperature_avg': (20, 35),
                'rainfall_annual': (1000, 2000),
                'rainfall_seasonal': (600, 1000),
                'humidity_avg': (70, 90),
                'sunshine_hours': (6, 8),
                'growing_season': (150, 150)
            },
            'wheat': {
                'temperature_avg': (15, 25),
                'rainfall_annual': (400, 800),
                'rainfall_seasonal': (200, 400),
                'humidity_avg': (40, 70),
                'sunshine_hours': (8, 10),
                'growing_season': (180, 180)
            },
            'sorghum': {
                'temperature_avg': (20, 35),
                'rainfall_annual': (300, 800),
                'rainfall_seasonal': (200, 500),
                'humidity_avg': (30, 70),
                'sunshine_hours': (8, 12),
                'growing_season': (120, 120)
            },
            'cassava': {
                'temperature_avg': (20, 30),
                'rainfall_annual': (800, 1500),
                'rainfall_seasonal': (400, 800),
                'humidity_avg': (60, 80),
                'sunshine_hours': (6, 10),
                'growing_season': (300, 300)
            },
            'yam': {
                'temperature_avg': (20, 30),
                'rainfall_annual': (1000, 1800),
                'rainfall_seasonal': (500, 1000),
                'humidity_avg': (60, 80),
                'sunshine_hours': (6, 8),
                'growing_season': (240, 240)
            }
        }
    
    def _define_climate_zones(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Define climate zone classifications."""
        return {
            'tropical_rainforest': {
                'temperature_avg': (24, 28),
                'rainfall_annual': (2000, 4000),
                'humidity_avg': (80, 95)
            },
            'tropical_savanna': {
                'temperature_avg': (22, 30),
                'rainfall_annual': (500, 1500),
                'humidity_avg': (50, 80)
            },
            'tropical_monsoon': {
                'temperature_avg': (20, 32),
                'rainfall_annual': (1000, 3000),
                'humidity_avg': (60, 90)
            },
            'semi_arid': {
                'temperature_avg': (20, 35),
                'rainfall_annual': (200, 600),
                'humidity_avg': (30, 60)
            },
            'arid': {
                'temperature_avg': (25, 40),
                'rainfall_annual': (0, 250),
                'humidity_avg': (10, 40)
            },
            'temperate': {
                'temperature_avg': (10, 20),
                'rainfall_annual': (600, 1200),
                'humidity_avg': (60, 80)
            }
        }
    
    def get_weather_data(self, latitude: float, longitude: float, 
                        days: int = 30) -> Dict[str, Any]:
        """
        Get current weather data from OpenWeatherMap API.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            days: Number of days to forecast
            
        Returns:
            Dictionary with weather data
        """
        if not self.api_key:
            print("API key not provided. Using sample data.")
            return self._get_sample_weather_data()
        
        try:
            # Current weather
            current_url = f"http://api.openweathermap.org/data/2.5/weather"
            current_params = {
                'lat': latitude,
                'lon': longitude,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            current_response = requests.get(current_url, params=current_params)
            current_data = current_response.json()
            
            # 5-day forecast
            forecast_url = f"http://api.openweathermap.org/data/2.5/forecast"
            forecast_params = {
                'lat': latitude,
                'lon': longitude,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            forecast_response = requests.get(forecast_url, params=forecast_params)
            forecast_data = forecast_response.json()
            
            return {
                'current': current_data,
                'forecast': forecast_data,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return self._get_sample_weather_data()
    
    def _get_sample_weather_data(self) -> Dict[str, Any]:
        """Get sample weather data for testing."""
        return {
            'current': {
                'main': {
                    'temp': 28.5,
                    'humidity': 65,
                    'pressure': 1013
                },
                'wind': {'speed': 3.2},
                'clouds': {'all': 40},
                'weather': [{'description': 'partly cloudy'}]
            },
            'forecast': {
                'list': [
                    {
                        'main': {'temp': 29.0, 'humidity': 70},
                        'wind': {'speed': 2.8},
                        'dt_txt': '2024-01-01 12:00:00'
                    }
                ]
            },
            'status': 'sample_data'
        }
    
    def analyze_climate_parameters(self, climate: ClimateParameters) -> Dict[str, Dict[str, Any]]:
        """
        Analyze climate parameters against optimal ranges.
        
        Args:
            climate: ClimateParameters object
            
        Returns:
            Dictionary with climate analysis results
        """
        analysis = {}
        
        for param, value in climate.__dict__.items():
            if param in self.optimal_climate_ranges:
                optimal_min, optimal_max = self.optimal_climate_ranges[param]
                
                if optimal_min <= value <= optimal_max:
                    status = "Optimal"
                    score = 100
                elif value < optimal_min:
                    status = "Below Optimal"
                    score = max(0, (value / optimal_min) * 100)
                else:
                    status = "Above Optimal"
                    score = max(0, 100 - ((value - optimal_max) / optimal_max) * 100)
                
                analysis[param] = {
                    'value': value,
                    'optimal_range': (optimal_min, optimal_max),
                    'status': status,
                    'score': round(score, 2),
                    'recommendation': self._get_climate_recommendation(param, value, optimal_min, optimal_max)
                }
        
        return analysis
    
    def _get_climate_recommendation(self, param: str, value: float, 
                                  optimal_min: float, optimal_max: float) -> str:
        """Get recommendation for climate parameter management."""
        recommendations = {
            'temperature_avg': {
                'low': "Consider using heat-tolerant varieties or greenhouse cultivation.",
                'high': "Use shade crops, irrigation cooling, or heat-resistant varieties."
            },
            'rainfall_annual': {
                'low': "Implement irrigation systems and water conservation practices.",
                'high': "Ensure proper drainage and use flood-tolerant crops."
            },
            'humidity_avg': {
                'low': "Increase irrigation frequency and use mulching to retain moisture.",
                'high': "Improve ventilation and use disease-resistant varieties."
            },
            'sunshine_hours': {
                'low': "Consider shade-tolerant crops or artificial lighting.",
                'high': "Use shade nets or intercropping to protect sensitive crops."
            }
        }
        
        if param in recommendations:
            if value < optimal_min:
                return recommendations[param]['low']
            elif value > optimal_max:
                return recommendations[param]['high']
        
        return "Climate conditions are within optimal range for most crops."
    
    def calculate_climate_suitability_score(self, climate: ClimateParameters) -> Dict[str, Any]:
        """
        Calculate overall climate suitability score.
        
        Args:
            climate: ClimateParameters object
            
        Returns:
            Dictionary with climate suitability assessment
        """
        analysis = self.analyze_climate_parameters(climate)
        
        # Weight different parameters
        weights = {
            'temperature_avg': 0.25,
            'rainfall_annual': 0.25,
            'humidity_avg': 0.15,
            'sunshine_hours': 0.15,
            'temperature_min': 0.10,
            'temperature_max': 0.10
        }
        
        weighted_score = 0
        total_weight = 0
        
        for param, data in analysis.items():
            if param in weights:
                weighted_score += data['score'] * weights[param]
                total_weight += weights[param]
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine suitability level
        if overall_score >= 80:
            suitability_level = "Excellent"
        elif overall_score >= 60:
            suitability_level = "Good"
        elif overall_score >= 40:
            suitability_level = "Moderate"
        else:
            suitability_level = "Poor"
        
        return {
            'overall_score': round(overall_score, 2),
            'suitability_level': suitability_level,
            'parameter_analysis': analysis,
            'recommendations': self._get_overall_climate_recommendations(analysis, overall_score)
        }
    
    def _get_overall_climate_recommendations(self, analysis: Dict, overall_score: float) -> List[str]:
        """Get overall climate recommendations."""
        recommendations = []
        
        if overall_score < 40:
            recommendations.append("CRITICAL: Climate conditions are challenging for agriculture.")
            recommendations.append("Consider: Greenhouses, irrigation systems, or alternative crops.")
        elif overall_score < 60:
            recommendations.append("MODERATE: Climate conditions require careful management.")
            recommendations.append("Focus on: Water management, crop selection, and timing.")
        elif overall_score < 80:
            recommendations.append("GOOD: Climate conditions are suitable with proper management.")
            recommendations.append("Maintain: Current practices and monitor for changes.")
        else:
            recommendations.append("EXCELLENT: Climate conditions are optimal for agriculture.")
            recommendations.append("Continue: Current practices and maximize potential.")
        
        # Add specific recommendations for low-scoring parameters
        low_params = [param for param, data in analysis.items() if data['score'] < 60]
        if low_params:
            recommendations.append(f"Priority climate factors to address: {', '.join(low_params[:3])}")
        
        return recommendations
    
    def classify_climate_zone(self, climate: ClimateParameters) -> str:
        """
        Classify climate zone based on temperature and rainfall.
        
        Args:
            climate: ClimateParameters object
            
        Returns:
            Climate zone classification
        """
        temp_avg = climate.temperature_avg
        rainfall = climate.rainfall_annual
        humidity = climate.humidity_avg
        
        # Check against climate zone definitions
        for zone, ranges in self.climate_zones.items():
            temp_range = ranges['temperature_avg']
            rain_range = ranges['rainfall_annual']
            hum_range = ranges['humidity_avg']
            
            if (temp_range[0] <= temp_avg <= temp_range[1] and
                rain_range[0] <= rainfall <= rain_range[1] and
                hum_range[0] <= humidity <= hum_range[1]):
                return zone.replace('_', ' ').title()
        
        return "Unclassified"
    
    def assess_crop_climate_suitability(self, climate: ClimateParameters, crop: str) -> Dict[str, Any]:
        """
        Assess climate suitability for specific crops.
        
        Args:
            climate: ClimateParameters object
            crop: Crop name
            
        Returns:
            Dictionary with crop climate suitability assessment
        """
        if crop not in self.crop_climate_requirements:
            return {"error": f"Crop '{crop}' not in database"}
        
        requirements = self.crop_climate_requirements[crop]
        climate_analysis = self.analyze_climate_parameters(climate)
        
        suitability_scores = []
        limiting_factors = []
        
        for param, requirement in requirements.items():
            if param == 'growing_season':  # Skip non-climate parameters
                continue
                
            if isinstance(requirement, tuple) and len(requirement) == 2:
                min_req, max_req = requirement
            else:
                continue  # Skip if not a tuple
                
            if param in climate_analysis:
                climate_value = climate_analysis[param]['value']
                
                if min_req <= climate_value <= max_req:
                    score = 100
                elif climate_value < min_req:
                    score = (climate_value / min_req) * 100
                    limiting_factors.append(f"{param} too low ({climate_value:.1f} < {min_req:.1f})")
                else:
                    score = max(0, 100 - ((climate_value - max_req) / max_req) * 100)
                    limiting_factors.append(f"{param} too high ({climate_value:.1f} > {max_req:.1f})")
                
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
            'recommendations': self._get_crop_climate_recommendations(crop, limiting_factors)
        }
    
    def _get_crop_climate_recommendations(self, crop: str, limiting_factors: List[str]) -> List[str]:
        """Get crop-specific climate recommendations."""
        recommendations = []
        
        crop_recs = {
            'maize': "For maize: Ensure adequate moisture during flowering, manage temperature stress.",
            'rice': "For rice: Maintain water levels, ensure warm temperatures, good drainage.",
            'wheat': "For wheat: Cool temperatures preferred, adequate moisture during growth.",
            'sorghum': "For sorghum: Drought-tolerant, warm temperatures, moderate rainfall.",
            'cassava': "For cassava: Warm temperatures, adequate rainfall, well-drained soil.",
            'yam': "For yam: Warm temperatures, high rainfall, good drainage, long growing season."
        }
        
        if crop in crop_recs:
            recommendations.append(crop_recs[crop])
        
        if limiting_factors:
            recommendations.append("Address limiting factors: " + ", ".join(limiting_factors[:2]))
        
        return recommendations
    
    def create_climate_report(self, climate: ClimateParameters, crops: Optional[List[str]] = None) -> str:
        """
        Create a comprehensive climate analysis report.
        
        Args:
            climate: ClimateParameters object
            crops: List of crops to assess suitability for
            
        Returns:
            Formatted climate analysis report
        """
        if crops is None:
            crops = ['maize', 'rice', 'wheat', 'sorghum']
        
        # Get climate zone
        climate_zone = self.classify_climate_zone(climate)
        
        # Get climate suitability analysis
        climate_suitability = self.calculate_climate_suitability_score(climate)
        
        # Get crop climate suitability assessments
        crop_assessments = {}
        for crop in crops:
            crop_assessments[crop] = self.assess_crop_climate_suitability(climate, crop)
        
        # Create report
        report = f"""
CLIMATE ANALYSIS REPORT
{'='*50}

CLIMATE CHARACTERISTICS:
   Climate Zone: {climate_zone}
   Average Temperature: {climate.temperature_avg:.1f}°C
   Annual Rainfall: {climate.rainfall_annual:.0f} mm
   Average Humidity: {climate.humidity_avg:.1f}%
   Sunshine Hours: {climate.sunshine_hours:.1f} hours/day
   Elevation: {climate.elevation:.0f} m

CLIMATE SUITABILITY ASSESSMENT:
   Overall Score: {climate_suitability['overall_score']}/100
   Suitability Level: {climate_suitability['suitability_level']}

RECOMMENDATIONS:
"""
        
        for rec in climate_suitability['recommendations']:
            report += f"   {rec}\n"
        
        report += f"\nCROP CLIMATE SUITABILITY ANALYSIS:\n"
        for crop, assessment in crop_assessments.items():
            report += f"\n   {crop.upper()}: {assessment['suitability_level']} ({assessment['suitability_score']:.1f}/100)\n"
            if assessment['limiting_factors']:
                report += f"      Limiting factors: {', '.join(assessment['limiting_factors'][:2])}\n"
        
        report += f"\n{'='*50}\n"
        report += f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return report
    
    def get_historical_climate_data(self, latitude: float, longitude: float, 
                                  start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical climate data (mock implementation).
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with historical climate data
        """
        # This would typically connect to a climate data API
        # For now, we'll generate sample data
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic sample data based on location
        np.random.seed(int(latitude * longitude))
        
        data = []
        for date in date_range:
            # Seasonal variations
            day_of_year = date.timetuple().tm_yday
            seasonal_temp = 25 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            
            data.append({
                'date': date,
                'temperature_avg': seasonal_temp + np.random.normal(0, 2),
                'temperature_min': seasonal_temp - 5 + np.random.normal(0, 1),
                'temperature_max': seasonal_temp + 5 + np.random.normal(0, 1),
                'rainfall': max(0, np.random.exponential(2)),
                'humidity': 60 + np.random.normal(0, 10),
                'sunshine_hours': 8 + np.random.normal(0, 2),
                'wind_speed': 2 + np.random.exponential(1)
            })
        
        return pd.DataFrame(data)


def create_sample_climate_data() -> ClimateParameters:
    """Create sample climate data for testing."""
    return ClimateParameters(
        temperature_avg=26.5,
        temperature_min=22.0,
        temperature_max=32.0,
        rainfall_annual=1200.0,
        rainfall_seasonal=600.0,
        humidity_avg=70.0,
        sunshine_hours=8.5,
        wind_speed=3.0,
        elevation=150.0,
        latitude=6.5,
        longitude=-1.6
    )


if __name__ == "__main__":
    # Create sample climate data
    climate = create_sample_climate_data()
    
    # Initialize analyzer
    analyzer = ClimateAnalyzer()
    
    # Perform analysis
    print("Climate Analysis Demo")
    print("="*50)
    
    # Generate report
    report = analyzer.create_climate_report(climate, ['maize', 'rice', 'wheat', 'sorghum'])
    print(report)
    
    print("Climate analysis completed successfully!")
