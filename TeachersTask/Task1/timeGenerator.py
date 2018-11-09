import datetime
import numpy as np

dt = datetime.datetime(2015, 1, 1, 8, 0, 0)
end = datetime.datetime(2016, 1, 1, 7, 59, 59)
step = datetime.timedelta(minutes=5)

result = []

while dt < end:
    result.append(dt.strftime('%Y%m%d%H%M%S'))
    dt += step

TimeGenerated = np.asarray(result)
print("Shape of Time Generated: ", TimeGenerated.shape)
print(TimeGenerated)