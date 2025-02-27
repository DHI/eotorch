BAND_INDEX = {
    "sentinel-2": {
        "bandmap" : {
            "coastal": 0,
            "blue": 1,
            "green": 2,
            "red": 3,
            "vnir1": 4,
            "vnir2": 5,
            "vnir3": 6,
            "vnir4": 7,
            "vnir5": 8,  # 8a in the documentation and subsequent numbers have been increased by 1
            "swir1": 9,
            "swir2": 10,
            "swir3": 11,
            "swir4": 12,
        },
        "res" : 10,
        
    },
    "sentinel-2-bathy": {
        "bandmap" : {
            "coastal": 0,
            "blue": 1,
            "green": 2,
            "red": 3,
            "nir": 4,
        },
        "res" : 10,

    },
    "landsat5": {
        "bandmap" : {
            "blue": 0,
            "green": 1,
            "red": 2,
            "nir": 3,
            "swir1": 4,
            "swir2": 5,
        },
        "res" : 30,
        
    },
    # VHR sensors
    "pneo": {
        "bandmap" : {
            "red": 0,
            "green": 1,
            "blue": 2,
            "nir": 3,
            "rededge": 4,
            "deepblue": 5,
        },
        "res" : 0.3, 
    },
    "phr": {
        "bandmap" : {
            "red": 0,
            "green": 1,
            "blue": 2,
            "nir": 3,
        },
        "res" : 0.5,
    },
    "wv": {
        "bandmap" : {
            "coastal": 0,
            "blue": 1,
            "green": 2,
            "yellow": 3,
            "red": 4,
            "rededge": 5,
            "nir1": 6,
            "nir2": 7,
        },
        "res" : 0.5,

    },
    # 4-band / Aerial
    "rgb": {
        "bandmap" : {
            "red": 0,
            "green": 1,
            "blue": 2,
        },
        "res" : 0.125,
    },
    "bgr": {
        "bandmap" : {
            "blue": 0,
            "green": 1,
            "red": 2,
        },
        "res" : 0.125,
    },
    "rgbi": {
        "bandmap" : {
            "red": 0,
            "green": 1,
            "blue": 2,
            "nir": 3,
        },
        "res" : 0.125,
    },
    "bgri": {
        "bandmap" : {
            "blue": 0,
            "green": 1,
            "red": 2,
            "nir": 3,
        },
        "res" : 0.125,
    },
}
