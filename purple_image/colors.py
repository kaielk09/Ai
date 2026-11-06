"""Lightweight prompt color parsing for Purple AI Image.

This is deterministic prompt assistance, not a claim that an untrained model
has learned every color. Arbitrary HEX and RGB colors are accepted.
"""
import colorsys
import re

NAMED = {
    "czerwony": "#E53935", "red": "#E53935",
    "pomarańczowy": "#FB8C00", "orange": "#FB8C00",
    "żółty": "#FDD835", "yellow": "#FDD835",
    "zielony": "#43A047", "green": "#43A047",
    "turkusowy": "#00A6A6", "turquoise": "#00A6A6",
    "cyjan": "#00BCD4", "cyan": "#00BCD4",
    "niebieski": "#1E88E5", "blue": "#1E88E5",
    "granatowy": "#172554", "navy": "#172554",
    "fioletowy": "#8E44AD", "purple": "#8E44AD",
    "różowy": "#EC407A", "pink": "#EC407A",
    "magenta": "#D81B60", "purpurowy": "#8E245C",
    "brązowy": "#795548", "brown": "#795548",
    "beżowy": "#D7C3A3", "beige": "#D7C3A3",
    "biały": "#FFFFFF", "white": "#FFFFFF",
    "czarny": "#111111", "black": "#111111",
    "szary": "#808080", "grey": "#808080", "gray": "#808080",
    "złoty": "#D4AF37", "gold": "#D4AF37",
    "srebrny": "#C0C0C0", "silver": "#C0C0C0",
    "miętowy": "#98FFCC", "mint": "#98FFCC",
    "lawendowy": "#B57EDC", "lavender": "#B57EDC",
    "koralowy": "#FF7F50", "coral": "#FF7F50",
    "oliwkowy": "#808000", "olive": "#808000",
    "bordowy": "#800020", "burgundy": "#800020",
    "szmaragdowy": "#50C878", "emerald": "#50C878",
    "pastelowy": "#D8BFD8", "pastel": "#D8BFD8",
}

def _hex_to_rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[i:i+2], 16) for i in (0, 2, 4))

def _rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*(max(0, min(255, round(v))) for v in rgb))

def _harmonies(hex_color):
    r, g, b = (v / 255 for v in _hex_to_rgb(hex_color))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    colors = []
    for shift, sat_mul, val_mul in ((0, .78, 1.0), (30/360, .72, .92),
                                    (-30/360, .68, 1.05), (180/360, .62, .88)):
        rr, gg, bb = colorsys.hsv_to_rgb((h + shift) % 1, min(1, s * sat_mul), min(1, v * val_mul))
        colors.append(_rgb_to_hex((rr*255, gg*255, bb*255)))
    return colors

def analyze_colors(prompt):
    text = str(prompt or "")
    found = []
    for match in re.findall(r"#[0-9a-fA-F]{6}\b", text):
        found.append(match.upper())
    for match in re.findall(r"(?i)\brgb\s*\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)", text):
        vals = tuple(int(x) for x in match)
        if all(0 <= x <= 255 for x in vals):
            found.append(_rgb_to_hex(vals))
    lowered = text.casefold()
    for name, value in NAMED.items():
        if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", lowered):
            found.append(value)
    unique = list(dict.fromkeys(found))
    wants_palette = any(word in lowered for word in (
        "odcien", "odcieni", "kolor", "barw", "fajny", "ładny", "ladny",
        "harmon", "pasuj", "shades", "colors", "palette", "nice"
    ))
    dark_request = any(word in lowered for word in ("ciemny", "dark", "mroczny"))
    bright_request = any(word in lowered for word in ("jasny", "light", "bright"))
    # When the user asks for attractive shades but gives no exact hue,
    # choose a restrained default palette rather than returning no guidance.
    if not unique and wants_palette:
        unique = ["#25263A", "#7656A6"] if dark_request else ["#53658C", "#9A78B8"]
        if bright_request:
            unique = ["#E7C98B", "#8AA7C8"]
    if not unique:
        return {"colors": [], "palette": [], "conditioning": ""}
    palette = []
    for color in unique[:4]:
        palette.extend(_harmonies(color))
    palette = list(dict.fromkeys(palette))[:12]
    style = []
    if any(x in lowered for x in ("ciemny", "dark", "mroczny")):
        style.append("dark tonal balance, controlled deep shadows")
    if any(x in lowered for x in ("jasny", "light", "bright")):
        style.append("bright highlights and luminous midtones")
    if any(x in lowered for x in ("pastel", "delikat", "soft")):
        style.append("soft pastel saturation")
    if any(x in lowered for x in ("neon", "żywy", "zywy", "vivid")):
        style.append("vivid high-chroma accents")
    conditioning = (
        "Color direction: main colors " + ", ".join(unique[:4]) +
        "; coordinated palette " + ", ".join(palette) +
        ("; " + ", ".join(style) if style else "") +
        ". Preserve the requested subject and composition."
    )
    return {"colors": unique[:4], "palette": palette, "conditioning": conditioning}
