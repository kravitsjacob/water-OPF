import pandas as pd


def generator_match(df_gen_info):
    # This is done manually through remotely sources images and spatial analysis
    manual_dict = {0: {'MATPOWER Index': 49,
             'EIA Plant Name': 'Rantoul',
             'Match Type': 'Location, Capacity',
             'POWERWORLD PLANT NAME': 'RANTOUL 2'},
         1: {'MATPOWER Index': 50,
             'EIA Plant Name': 'Rantoul',
             'Match Type': 'Location, Capacity',
             'POWERWORLD PLANT NAME': 'RANTOUL 2'},
         2: {'MATPOWER Index': 51,
             'EIA Plant Name': 'Rantoul',
             'Match Type': 'Location, Capacity',
             'POWERWORLD PLANT NAME': 'RANTOUL 2'},
         3: {'MATPOWER Index': 52,
             'EIA Plant Name': 'Rantoul',
             'Match Type': 'Location, Capacity',
             'POWERWORLD PLANT NAME': 'RANTOUL 2'},
         4: {'MATPOWER Index': 53,
             'EIA Plant Name': 'Rantoul',
             'Match Type': 'Location, Capacity',
             'POWERWORLD PLANT NAME': 'RANTOUL 2'},
         5: {'MATPOWER Index': 65,
             'EIA Plant Name': 'Pioneer Trail Wind Farm, LLC',
             'Match Type': 'Location, Capacity, Fuel Type',
             'POWERWORLD PLANT NAME': 'PAXTON 1'},
         6: {'MATPOWER Index': 67,
             'EIA Plant Name': 'Archer Daniels Midland Decatur',
             'Match Type': 'Location, Capacity, Fuel Type',
             'POWERWORLD PLANT NAME': 'MOUNT ZION'},
         7: {'MATPOWER Index': 68,
             'EIA Plant Name': 'Archer Daniels Midland Decatur',
             'Match Type': 'Location, Capacity, Fuel Type',
             'POWERWORLD PLANT NAME': 'MOUNT ZION'},
         8: {'MATPOWER Index': 69,
             'EIA Plant Name': 'Archer Daniels Midland Decatur',
             'Match Type': 'Location, Capacity, Fuel Type',
             'POWERWORLD PLANT NAME': 'MOUNT ZION'},
         9: {'MATPOWER Index': 70,
             'EIA Plant Name': 'Archer Daniels Midland Decatur',
             'Match Type': 'Location, Capacity, Fuel Type',
             'POWERWORLD PLANT NAME': 'MOUNT ZION'},
         10: {'MATPOWER Index': 71,
              'EIA Plant Name': 'Archer Daniels Midland Decatur',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'MOUNT ZION'},
         11: {'MATPOWER Index': 72,
              'EIA Plant Name': 'Archer Daniels Midland Decatur',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'MOUNT ZION'},
         12: {'MATPOWER Index': 73,
              'EIA Plant Name': 'Archer Daniels Midland Decatur',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'MOUNT ZION'},
         13: {'MATPOWER Index': 76,
              'EIA Plant Name': 'Archer Daniels Midland Decatur',
              'Match Type': 'Location, Capacity',
              'POWERWORLD PLANT NAME': 'BRIMFIELD'},
         14: {'MATPOWER Index': 77,
              'EIA Plant Name': 'Archer Daniels Midland Decatur',
              'Match Type': 'Location, Capacity',
              'POWERWORLD PLANT NAME': 'BRIMFIELD'},
         15: {'MATPOWER Index': 78,
              'EIA Plant Name': 'Archer Daniels Midland Decatur',
              'Match Type': 'Location, Capacity',
              'POWERWORLD PLANT NAME': 'BRIMFIELD'},
         16: {'MATPOWER Index': 79,
              'EIA Plant Name': 'Archer Daniels Midland Decatur',
              'Match Type': 'Location, Capacity',
              'POWERWORLD PLANT NAME': 'BRIMFIELD'},
         17: {'MATPOWER Index': 90,
              'EIA Plant Name': 'Clinton LFGTE',
              'Match Type': 'Location, Capacity',
              'POWERWORLD PLANT NAME': 'CLINTON 3'},
         18: {'MATPOWER Index': 91,
              'EIA Plant Name': 'Clinton LFGTE',
              'Match Type': 'Location, Capacity',
              'POWERWORLD PLANT NAME': 'CLINTON 3'},
         19: {'MATPOWER Index': 92,
              'EIA Plant Name': 'Clinton LFGTE',
              'Match Type': 'Location, Capacity',
              'POWERWORLD PLANT NAME': 'CLINTON 3'},
         20: {'MATPOWER Index': 94,
              'EIA Plant Name': 'Tuscola Station',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'TUSCOLA 2'},
         21: {'MATPOWER Index': 104,
              'EIA Plant Name': 'High Trail Wind Farm LLC',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'ELLSWORTH 1'},
         22: {'MATPOWER Index': 105,
              'EIA Plant Name': 'High Trail Wind Farm LLC',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'ELLSWORTH 1'},
         23: {'MATPOWER Index': 114,
              'EIA Plant Name': 'White Oak Energy LLC',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'NORMAL 2'},
         24: {'MATPOWER Index': 115,
              'EIA Plant Name': 'White Oak Energy LLC',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'NORMAL 2'},
         25: {'MATPOWER Index': 125,
              'EIA Plant Name': 'E D Edwards',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'BARTONVILLE'},
         26: {'MATPOWER Index': 126,
              'EIA Plant Name': 'E D Edwards',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'BARTONVILLE'},
         27: {'MATPOWER Index': 127,
              'EIA Plant Name': 'E D Edwards',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'BARTONVILLE'},
         28: {'MATPOWER Index': 135,
              'EIA Plant Name': 'Powerton',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'PEKIN 1'},
         29: {'MATPOWER Index': 136,
              'EIA Plant Name': 'Powerton',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'PEKIN 1'},
         30: {'MATPOWER Index': 147,
              'EIA Plant Name': 'Rail Splitter Wind Farm',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'HOPEDALE 2'},
         31: {'MATPOWER Index': 151,
              'EIA Plant Name': 'Dallman',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'SPRINGFIELD 5'},
         32: {'MATPOWER Index': 152,
              'EIA Plant Name': 'Dallman',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'SPRINGFIELD 5'},
         33: {'MATPOWER Index': 153,
              'EIA Plant Name': 'Dallman',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'SPRINGFIELD 5'},
         34: {'MATPOWER Index': 154,
              'EIA Plant Name': 'Dallman',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'SPRINGFIELD 5'},
         35: {'MATPOWER Index': 155,
              'EIA Plant Name': 'Dallman',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'SPRINGFIELD 5'},
         36: {'MATPOWER Index': 161,
              'EIA Plant Name': 'Interstate',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'SPRINGFIELD 4'},
         37: {'MATPOWER Index': 164,
              'EIA Plant Name': 'University of Illinois Abbott Power Plt',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'CHAMPAIGN 1'},
         38: {'MATPOWER Index': 165,
              'EIA Plant Name': 'University of Illinois Abbott Power Plt',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'CHAMPAIGN 1'},
         39: {'MATPOWER Index': 166,
              'EIA Plant Name': 'University of Illinois Abbott Power Plt',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'CHAMPAIGN 1'},
         40: {'MATPOWER Index': 167,
              'EIA Plant Name': 'University of Illinois Abbott Power Plt',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'CHAMPAIGN 1'},
         41: {'MATPOWER Index': 168,
              'EIA Plant Name': 'University of Illinois Abbott Power Plt',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'CHAMPAIGN 1'},
         42: {'MATPOWER Index': 169,
              'EIA Plant Name': 'University of Illinois Abbott Power Plt',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'CHAMPAIGN 1'},
         43: {'MATPOWER Index': 170,
              'EIA Plant Name': 'University of Illinois Abbott Power Plt',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'CHAMPAIGN 1'},
         44: {'MATPOWER Index': 182,
              'EIA Plant Name': 'Dallman',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'SPRINGFIELD 2'},
         45: {'MATPOWER Index': 183,
              'EIA Plant Name': 'Dallman',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'SPRINGFIELD 2'},
         46: {'MATPOWER Index': 189,
              'EIA Plant Name': 'Clinton Power Station',
              'Match Type': 'Location, Capacity',
              'POWERWORLD PLANT NAME': 'CLINTON 1'},
         47: {'MATPOWER Index': 196,
              'EIA Plant Name': 'Gibson City Energy Center LLC',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'GIBSON CITY 1'},
         48: {'MATPOWER Index': 197,
              'EIA Plant Name': 'Gibson City Energy Center LLC',
              'Match Type': 'Location, Capacity, Fuel Type',
              'POWERWORLD PLANT NAME': 'GIBSON CITY 1'}}
    df_matches = pd.DataFrame.from_records(manual_dict).T

    # Merge manual matches
    df_gen_info_match = df_gen_info.merge(df_matches)
    return df_gen_info_match