BAND_INDEX = {
  'sentinel-2' : {
    'coastal' : 0,
    'blue' : 1,
    'green' : 2,
    'red' : 3,
    'vnir1' : 4,
    'vnir2' : 5, 
    'vnir3' : 6,
    'vnir4' : 7,
    'vnir5' : 8, # 8a in the documentation and subsequent numbers have been increased by 1
    'swir1' : 9,
    'swir2' : 10,
    'swir3' : 11,
    'swir4' : 12,
  },
  'sentinel-2-bathy' : {
    'coastal' : 0,
    'blue' : 1,
    'green' : 2,
    'red' : 3,
    'nir' : 4,
  },
  'landsat5' : {
    'blue' : 0,
    'green' : 1,
    'red' : 2,
    'nir' : 3,
    'swir1' : 4,
    'swir2' : 5,
  },
  
  # VHR sensors
  'pneo' : {
    'red' : 0,
    'green' : 1,
    'blue' : 2,
    'nir' : 3,
    'rededge' : 4,
    'deepblue' : 5,
  },
  'phr' : {
    'red' : 0,
    'green' : 1,
    'blue' : 2,
    'nir' : 3,
  },
  'wv' : {
    'coastal' : 0,
    'blue' : 1,
    'green' : 2,
    'yellow' : 3,
    'red' : 4,
    'rededge' : 5,
    'nir1' : 6,
    'nir2' : 7,
  },
  # 4-band / Aerial
  'rgb' : {
    'red' : 0,
    'green' : 1,
    'blue' : 2,
  },
  'bgr' : {
    'blue' : 0,
    'green' : 1,
    'red' : 2,
  },
  'rgbi' : {
    'red' : 0,
    'green' : 1,
    'blue' : 2,
    'nir' : 3,
  },
  'bgri' : {
    'blue' : 0,
    'green' : 1,
    'red' : 2,
    'nir' : 3,
  }
}