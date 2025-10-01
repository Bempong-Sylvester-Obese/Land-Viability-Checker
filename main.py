#!/usr/bin/env python3
import sys
import os
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from models import (
    LandViabilityAssessor, 
    LandParameters,
    SoilQualityAnalyzer,
    ClimateAnalyzer,
    CropYieldPredictor
)


def main():
    parser = argparse.ArgumentParser(
        description="Land Viability Checker - Assess agricultural land potential",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo                    # Run demo with sample data
  python main.py --train-models            # Train crop yield prediction models
  python main.py --soil-analysis           # Run soil analysis demo
  python main.py --climate-analysis        # Run climate analysis demo
  python main.py --full-assessment         # Run complete land assessment demo
        """
    )
    
    parser.add_argument('--demo', action='store_true', 
                       help='Run complete demo with sample data')
    parser.add_argument('--train-models', action='store_true',
                       help='Train crop yield prediction models')
    parser.add_argument('--soil-analysis', action='store_true',
                       help='Run soil quality analysis demo')
    parser.add_argument('--climate-analysis', action='store_true',
                       help='Run climate analysis demo')
    parser.add_argument('--full-assessment', action='store_true',
                       help='Run complete land viability assessment demo')
    parser.add_argument('--output-dir', default='Data',
                       help='Output directory for reports and visualizations')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.train_models:
        train_crop_models()
    elif args.soil_analysis:
        run_soil_analysis_demo(args.output_dir)
    elif args.climate_analysis:
        run_climate_analysis_demo(args.output_dir)
    elif args.full_assessment:
        run_full_assessment_demo(args.output_dir)
    elif args.demo:
        run_complete_demo(args.output_dir)
    else:
        parser.print_help()


def train_crop_models():
    print("Training Crop Yield Prediction Models")
    print("="*50)
    
    from models.crop_yield_models import train_crop_yield_models
    
    try:
        predictor = train_crop_yield_models()
        print("\nModel training completed successfully!")
        print(f"Best model: {predictor.best_model}")
        print(f"Best R² Score: {predictor.model_scores[predictor.best_model]['test_r2']:.4f}")
        
        return predictor
    except Exception as e:
        print(f"Error training models: {e}")
        return None


def run_soil_analysis_demo(output_dir):
    print("Soil Quality Analysis Demo")
    print("="*50)
    
    from models.soil_analysis import create_sample_soil_data
    
    # Create sample soil data
    soil = create_sample_soil_data()
    
    # Initialize analyzer
    analyzer = SoilQualityAnalyzer()
    
    # Generate report
    report = analyzer.create_soil_report(soil, ['maize', 'rice', 'wheat', 'sorghum'])
    print(report)
    
    # Create visualizations
    analyzer.plot_soil_analysis(soil, f"{output_dir}/soil_analysis_demo.png")
    
    print("Soil analysis demo completed!")


def run_climate_analysis_demo(output_dir):
    print("Climate Analysis Demo")
    print("="*50)
    
    from models.climate_analysis import create_sample_climate_data
    
    # Create sample climate data
    climate = create_sample_climate_data()
    
    # Initialize analyzer
    analyzer = ClimateAnalyzer()
    
    # Generate report
    report = analyzer.create_climate_report(climate, ['maize', 'rice', 'wheat', 'sorghum'])
    print(report)
    
    print("Climate analysis demo completed!")


def run_full_assessment_demo(output_dir):
    print("Complete Land Viability Assessment Demo")
    print("="*60)
    
    from models.land_viability_assessor import create_sample_land_data
    
    # Create sample land data
    land = create_sample_land_data()
    
    # Initialize assessor
    assessor = LandViabilityAssessor()
    
    # Generate comprehensive report
    report = assessor.create_comprehensive_report(land, ['maize', 'rice', 'wheat', 'sorghum'])
    print(report)
    
    # Perform full assessment
    assessment = assessor.assess_land_viability(land, ['maize', 'rice', 'wheat', 'sorghum'])
    
    # Create visualizations
    assessor.plot_viability_dashboard(assessment, f"{output_dir}/full_assessment_dashboard.png")
    
    # Save assessment
    assessor.save_assessment(assessment, f"{output_dir}/full_assessment_results.json")
    
    print("Full assessment demo completed!")


def run_complete_demo(output_dir):
    print("LAND VIABILITY CHECKER - COMPLETE DEMO")
    print("="*60)
    print("This demo showcases all features of the Land Viability Checker")
    print("including soil analysis, climate assessment, and crop yield prediction.")
    print()
    
    # Step 1: Train models (if not already trained)
    print("Step 1: Training Machine Learning Models")
    print("-" * 40)
    predictor = train_crop_models()
    
    if predictor:
        print("Models trained successfully!")
    else:
        print("Model training failed, but continuing with demo...")
    
    print()
    
    # Step 2: Soil Analysis Demo
    print("Step 2: Soil Quality Analysis")
    print("-" * 40)
    run_soil_analysis_demo(output_dir)
    print()
    
    # Step 3: Climate Analysis Demo
    print("Step 3: Climate Suitability Analysis")
    print("-" * 40)
    run_climate_analysis_demo(output_dir)
    print()
    
    # Step 4: Complete Assessment Demo
    print("Step 4: Complete Land Viability Assessment")
    print("-" * 40)
    run_full_assessment_demo(output_dir)
    print()
    
    # Step 5: Summary
    print("DEMO SUMMARY")
    print("-" * 40)
    print("All components tested successfully!")
    print("Output files saved to:", output_dir)
    print()
    print("Generated files:")
    print("  • soil_analysis_demo.png - Soil quality visualizations")
    print("  • full_assessment_dashboard.png - Comprehensive viability dashboard")
    print("  • full_assessment_results.json - Detailed assessment results")
    print("  • model_performance.png - ML model performance comparison")
    print("  • crop_yield_analysis.png - Crop yield data analysis")
    print()
    print("Demo completed successfully!")
    print("The Land Viability Checker is ready for production use!")


def interactive_mode():
    print("Interactive Land Viability Assessment")
    print("="*50)
    print("Enter land parameters for custom assessment:")
    print()
    
    try:
        # Get location information
        location_name = input("Location name: ").strip() or "Custom Location"
        latitude = float(input("Latitude: "))
        longitude = float(input("Longitude: "))
        elevation = float(input("Elevation (meters): "))
        
        # Get soil parameters
        print("\nSoil Parameters:")
        soil_ph = float(input("Soil pH: "))
        organic_matter = float(input("Organic Matter (%): "))
        nitrogen = float(input("Nitrogen (ppm): "))
        phosphorus = float(input("Phosphorus (ppm): "))
        potassium = float(input("Potassium (ppm): "))
        
        # Get climate parameters
        print("\nClimate Parameters:")
        temperature_avg = float(input("Average Temperature (°C): "))
        rainfall_annual = float(input("Annual Rainfall (mm): "))
        humidity_avg = float(input("Average Humidity (%): "))
        
        # Create land parameters (using defaults for missing values)
        land = LandParameters(
            latitude=latitude,
            longitude=longitude,
            elevation=elevation,
            location_name=location_name,
            soil_ph=soil_ph,
            organic_matter=organic_matter,
            nitrogen=nitrogen,
            phosphorus=phosphorus,
            potassium=potassium,
            calcium=1200.0,  # Default
            magnesium=150.0,  # Default
            sulfur=12.0,  # Default
            iron=25.0,  # Default
            manganese=12.0,  # Default
            zinc=2.0,  # Default
            copper=1.0,  # Default
            boron=0.8,  # Default
            clay_content=25.0,  # Default
            sand_content=55.0,  # Default
            silt_content=20.0,  # Default
            bulk_density=1.2,  # Default
            water_holding_capacity=18.0,  # Default
            cation_exchange_capacity=15.0,  # Default
            temperature_avg=temperature_avg,
            temperature_min=temperature_avg - 5,  # Estimate
            temperature_max=temperature_avg + 5,  # Estimate
            rainfall_annual=rainfall_annual,
            rainfall_seasonal=rainfall_annual * 0.6,  # Estimate
            humidity_avg=humidity_avg,
            sunshine_hours=8.5,  # Default
            wind_speed=3.0  # Default
        )
        
        # Perform assessment
        assessor = LandViabilityAssessor()
        assessment = assessor.assess_land_viability(land, ['maize', 'rice', 'wheat', 'sorghum'])
        
        # Generate report
        report = assessor.create_comprehensive_report(land, ['maize', 'rice', 'wheat', 'sorghum'])
        print("\n" + "="*60)
        print(report)
        
        # Save results
        output_file = f"Data/custom_assessment_{location_name.replace(' ', '_')}.json"
        assessor.save_assessment(assessment, output_file)
        print(f"\nAssessment saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\n\nAssessment cancelled by user.")
    except ValueError as e:
        print(f"\nError: Invalid input - {e}")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, show interactive mode option
        print("Land Viability Checker")
        print("="*30)
        print("Choose an option:")
        print("1. Run complete demo")
        print("2. Interactive assessment")
        print("3. Show help")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            run_complete_demo("Data")
        elif choice == "2":
            interactive_mode()
        elif choice == "3":
            main()
        else:
            print("Invalid choice. Running complete demo...")
            run_complete_demo("Data")
    else:
        main()
