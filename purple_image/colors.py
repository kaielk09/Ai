"""Prompt color parsing for Purple AI Image.

Accepts arbitrary RGB/HEX colors plus CSS color names via Pillow.
This only builds text conditioning; it does not paint colors onto the output.
"""
import colorsys
import re
from PIL import ImageColor

# Polish aliases; all CSS4 names are also accepted by Pillow.
POLISH = {
    "czerwony": "#E53935", "pomarańczowy": "#FB8C00",
    "żółty": "#FDD835", "zielony": "#43A047",
    "turkusowy": "#00A6A6", "niebieski": "#1E88E5",
    "granatowy": "#172554", "fioletowy": "#8E44AD",
    "purpurowy": "#800080", "różowy": "#EC407A",
    "brązowy": "#795548", "beżowy": "#D7C3A3",
    "biały": "#FFFFFF", "czarny": "#111111", "szary": "#808080",
    "złoty": "#D4AF37", "srebrny": "#C0C0C0",
    "miętowy": "#98FFCC", "lawendowy": "#B57EDC",
    "koralowy": "#FF7F50", "oliwkowy": "#808000",
    "bordowy": "#800020", "szmaragdowy": "#50C878",
}

def _rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(
        *(max(0, min(255, round(v))) for v in rgb)
    )

def _harmonies(hex_color):
    r, g, b = (v / 255 for v in ImageColor.getrgb(hex_color)[:3])
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    result = []
    for shift, sat_mul, val_mul in (
        (0, .78, 1.0), (30/360, .72, .92),
        (-30/360, .68, 1.05), (180/360, .62, .88),
    ):
        rr, gg, bb = colorsys.hsv_to_rgb(
            (h + shift) % 1, min(1, s * sat_mul), min(1, v * val_mul)
        )
        result.append(_rgb_to_hex((rr * 255, gg * 255, bb * 255)))
    return result

def _extract_colors(text):
    found = []
    # #RGB, #RRGGBB and #RRGGBBAA (alpha is ignored for conditioning).
    for value in re.findall(r"(?i)#(?:[0-9a-f]{8}|[0-9a-f]{6}|[0-9a-f]{3})\b", text):
        try:
            rgb = ImageColor.getrgb(value)
            found.append(_rgb_to_hex(rgb[:3]))
        except ValueError:
            pass

    # RGB integer channels, including CSS rgb(10 20 30) and rgb(10,20,30).
    pattern = r"(?i)\brgb\s*\(\s*(\d{1,3})\s*[, ]\s*(\d{1,3})\s*[, ]\s*(\d{1,3})\s*\)"
    for match in re.findall(pattern, text):
        values = tuple(map(int, match))
        if all(0 <= value <= 255 for value in values):
            found.append(_rgb_to_hex(values))

    lowered = text.casefold()
    for name, value in POLISH.items():
        if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", lowered):
            found.append(value)

    # CSS named colors: e.g. rebeccapurple, cornflowerblue, darkslategray.
    # Check longest names first so multiword/compound names are deterministic.
    for name in sorted(ImageColor.colormap, key=len, reverse=True):
        if re.search(r"(?<![a-z])" + re.escape(name) + r"(?![a-z])", lowered):
            try:
                found.append(_rgb_to_hex(ImageColor.getrgb(name)[:3]))
            except ValueError:
                continue
    return list(dict.fromkeys(found))

def analyze_colors(prompt):
    text = str(prompt or "")
    unique = _extract_colors(text)
    lowered = text.casefold()

    wants_palette = any(word in lowered for word in (
        "odcień", "odcienie", "odcieni", "kolor", "barw", "harmonia",
        "paleta", "palette", "shades", "colors", "colour",
    ))
    dark_request = any(word in lowered for word in ("ciemny", "dark", "mroczny"))
    bright_request = any(word in lowered for word in ("jasny", "light", "bright"))
    pastel_request = any(word in lowered for word in ("pastel", "delikatny", "soft"))
    vivid_request = any(word in lowered for word in ("neon", "żywy", "zywy", "vivid"))

    if not unique and wants_palette:
        unique = ["#25263A", "#7656A6"] if dark_request else ["#53658C", "#9A78B8"]
        if bright_request:
            unique = ["#E7C98B", "#8AA7C8"]

    if not unique:
        return {"colors": [], "palette": [], "conditioning": ""}

    palette = []
    for color in unique[:8]:
        palette.extend(_harmonies(color))
    palette = list(dict.fromkeys(palette))[:24]

    style = []
    if dark_request:
        style.append("dark tonal balance and controlled deep shadows")
    if bright_request:
        style.append("bright highlights and luminous midtones")
    if pastel_request:
        style.append("soft pastel saturation")
    if vivid_request:
        style.append("vivid high-chroma accents")

    conditioning = (
        "Color direction: preserve these requested colors "
        + ", ".join(unique[:8])
        + "; coordinated shade palette "
        + ", ".join(palette)
        + (("; " + ", ".join(style)) if style else "")
        + ". Keep the subject and composition faithful to the prompt."
    )
    return {"colors": unique[:8], "palette": palette, "conditioning": conditioning}
