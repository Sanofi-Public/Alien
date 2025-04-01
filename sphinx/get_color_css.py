colors = {
    '2980b9': '7a00e6',
    '2e8ece': '9143f8',
    '409ad5': '9e6df2',
    '6ab0de': 'ac91ff',
}



with open("build/html/_static/css/theme.css", 'r') as f:
    file = f.read()

all_lines = [l + '}' for l in file.split('}')]

lines = [l.replace(c, colors[c]) for c in colors for l in all_lines if c in l]

with open('sanofi.css', 'w') as f:
    f.write('\n'.join(lines))
