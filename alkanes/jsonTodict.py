import json

with open('LinearAlkanes.json', 'r') as file:
    alkanes_data = json.load(file)

#convert the JSON data to a dictionary
# Handle the case where alkanes_data is a list
# alkanes_dict = {alkane['name']: alkane for alkane in alkanes_data}
#
# # Access properties like boiling point
# methane_boiling_point = alkanes_dict['Methane']['boiling_point']
# print(f'Methane Boiling Point: {methane_boiling_point}°C')

methane_density = alkanes_data['methane']['density']
print(f"Density of Methane: {methane_density}")