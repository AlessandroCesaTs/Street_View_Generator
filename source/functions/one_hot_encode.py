import torch

def one_hot_encode(country):
    countries_dict = {'Argentina': 0,'Australia': 1,'Austria': 2,'Belgium': 3,'Brazil': 4,
    'Bulgaria': 5,'Canada': 6,'Chile': 7,'Colombia': 8,'Czechia': 9,'Finland': 10,'France': 11,
    'Germany': 12,'Greece': 13,'Indonesia': 14,'Ireland': 15,'Israel': 16,'Italy': 17,
    'Japan': 18,'Malaysia': 19,'Mexico': 20,'Netherlands': 21,'New_Zealand': 22,
    'Norway': 23,'Peru': 24,'Philippines': 25,'Poland': 26,'Portugal': 27,'Romania': 28,
    'Russia': 29,'Singapore': 30,'South_Africa': 31,'South_Korea': 32,'Spain': 33,
    'Sweden': 34,'Taiwan': 35,'Thailand': 36,'Turkey': 37,'United_Kingdom': 38,'United_States': 39
}
    encoding=torch.zeros(len(countries_dict))
    index=countries_dict.get(country)
    if index is not None:
        encoding[index]=1
        return encoding
    else:
        raise ValueError(f"Country '{country}' is not in the countries list.")