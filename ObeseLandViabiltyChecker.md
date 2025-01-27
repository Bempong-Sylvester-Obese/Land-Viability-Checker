class ObeseViabilityChecker:
    def __init__(self, soil_ph, water_availability, sunlight_hours):
        self.soil_ph = soil_ph
        self.water_availability = water_availability
        self.sunlight_hours = sunlight_hours
# Checking soil pH
    def check_soil_ph(self):
        # Viable soil pH level is between 6.0 and 7.5
        if 6.0 <= self.soil_ph <= 7.5:
            return True
        else:
            return False
# checking water availability
    def check_water_availability(self):
        # Viable water availability is at least 500mm per year
        if self.water_availability >= 500:
            return True
        else:
            return False
# Sunlight Intensity and Duration
    def check_sunlight_hours(self):
        # Viable sunlight exposure is at least 6 hours per day
        if self.sunlight_hours >= 6:
            return True
        else:
            return False

    def is_viable(self):
        # Check all criteria
        if self.check_soil_ph() and self.check_water_availability() and self.check_sunlight_hours():
            return "The land is viable for agriculture."
        else:
            return "The land is not viable for agriculture."

# Example usage:
soil_ph = 6.8
water_availability = 600  # in mm per year
sunlight_hours = 7  # in hours per day

checker = ObeseViabilityChecker(soil_ph, water_availability, sunlight_hours)
result = checker.is_viable()
print(result)
