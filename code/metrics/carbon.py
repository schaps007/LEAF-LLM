"""
Carbon calculation utilities
"""

def calculate_carbon(energy_kwh, carbon_intensity=0.233):
    """
    Calculate carbon emissions from energy consumption
    
    Args:
        energy_kwh: Energy in kilowatt-hours
        carbon_intensity: Carbon intensity in kg CO2e per kWh (default: 0.233 for average US grid)
    
    Returns:
        Carbon emissions in kg CO2e
    """
    return energy_kwh * carbon_intensity

def energy_from_carbon(carbon_kg, carbon_intensity=0.233):
    """
    Estimate energy from carbon emissions
    
    Args:
        carbon_kg: Carbon emissions in kg CO2e
        carbon_intensity: Carbon intensity in kg CO2e per kWh
    
    Returns:
        Energy in kWh
    """
    return carbon_kg / carbon_intensity if carbon_intensity > 0 else 0