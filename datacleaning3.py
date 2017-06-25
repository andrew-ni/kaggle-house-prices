import numpy as np
import re

# initializes dictionaries needed to enumerate values
def initialize_dicts(filename):
  f = open(filename, 'r')
  headers = f.readline().rstrip('\n').split(',')[1:-1]  # list of headers, excluding id and saleprice
  data = list(f)
  f.close()

  header_values = dict()

  for header in headers:
    header_values[header] = dict()

  for line in data:
    line = line.rstrip('\n').split(',')[1:-1]
    for i in range(len(line)):
      if re.search('[a-zA-Z]', line[i]) and headers[i] not in ['LotFrontage', 'MasVnrType']:
        if line[i] not in header_values[headers[i]]:
          header_values[headers[i]][line[i]] = len(header_values[headers[i]])

  return header_values, headers

def read_file( filename ):
  header_values, headers = initialize_dicts(filename)

  features = list()
  target = list()
  f = open(filename, 'r')
  data = list(f)[1:]
  f.close()

  for line in data:
    line = line.rstrip('\n').split(',')[1:]
    new_line = list()
    for i in range(len(line)-1):
      if re.search('[a-zA-Z]', line[i]):
        if headers[i] in ['LotFrontage', 'MasVnrType']:
          new_line.append(0.0)
        else:
          encoding = [0.0] * len(header_values[headers[i]])
          encoding[header_values[headers[i]][line[i]]] = 1.0
          new_line.extend(list(encoding))
      else:
        new_line.append(float(line[i]))
    features.append(new_line)
    target.append(float(line[len(line)-1]))

  return np.array(features), np.array(target).reshape(len(target), 1)

