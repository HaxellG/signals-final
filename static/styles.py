LIGHT_BLUE_COLOR = '#90d6ff'
MEDIUM_BLUE_COLOR = '#2781ff'
DARK_BLUE_COLOR = '#225eb1'
LIGHT_PURPLE_COLOR = "#a353ff"
DARK_PURPLE_COLOR = '#7955ac'
LIGHT_PURPLE_BG = '#f3e6ff'
DARK_PURPLE_TEXT = '#8a2be2'
PURE_BLACK_COLOR = '#340d61'

def build_custom_error(text: str):
    return f"""
        <div style='background-color: {LIGHT_PURPLE_BG}; padding: 12px; border-radius: 8px; 
                    margin: 15px 0; text-align: center;'>
            <span style='color: {DARK_PURPLE_TEXT}; font-size: 16px; font-weight: bold;'>
                {text}
            </span>
        </div>
    """

CSS_TOPBAR_STYLES = f"""
    <style>
        [data-testid="stDecoration"] {{
            background-image: linear-gradient(to right, {DARK_BLUE_COLOR}, {DARK_PURPLE_COLOR}, {MEDIUM_BLUE_COLOR});
            height: 5px;  /* Ajusta el grosor de la línea si lo deseas */
        }}
    </style>
"""

CSS_SIDEBARD_STYLES = """
<style>
    /* Main content area */

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #2b5876;
        background-image: linear-gradient(to bottom, #2b5876, #4e4376);
        padding: 2rem 1rem;
    }

    /* Sidebar title (Operaciones de Convolución) */
    [data-testid="stSidebar"] h1 {
        color: #ffffff !important;  /* Set to white */
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 1.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }

    .stSelectbox label {
        color: white !important;
        font-size: 40px !important;
    }

    /* Section headers */
    [data-testid="stSidebar"] h2 {
        color: #ffffff !important;  /* Set to white */
        font-size: 1em;
        font-weight: bold;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        letter-spacing: 0.05em;
    }

    [data-testid="stSidebar"] .stRadio > label:hover {
        color: #ffd700 !important;
    }

    [data-testid="stSidebar"] .stRadio > div > label {
        display: flex;
        align-items: center;
        cursor: pointer;
        margin-bottom: 0.5rem;
    }

    [data-testid="stSidebar"] .stRadio > div > label > div:first-child {
        display: none;
    }

    [data-testid="stSidebar"] .stRadio > div > label > div:last-child {
        display: flex;
        align-items: center;
    }

    [data-testid="stSidebar"] .stRadio > div > label > div:last-child::before {
        content: "•";
        font-size: 1.5em;
        margin-right: 0.5rem;
        color: transparent;
        transition: color 0.3s ease;
    }

    [data-testid="stSidebar"] .stRadio > div > label:hover > div:last-child::before {
        color: #ffd700 !important;
    }

    [data-testid="stSidebar"] .stRadio > div [data-testid="stMarkdownContainer"] p {
        margin: 0;
        color: #ffffff !important;
    }

    /* Main content styling */
    .main .block-container {
        padding: 2rem;
    }

    .main h1 {
        color: #2b5876;
        margin-bottom: 1rem;
    }
</style>
"""

CSS_CREDITS_STYLES = f"""
    <style>
        .custom-header {{
            color: #3C3C3C;  /* Gris oscuro suave para los títulos */
            background-color: #D6EAF8;  /* Azul pastel claro */
            padding: 15px;
            border-radius: 10px;
            font-weight: bold;
            font-size: 1.2rem;
        }}
        .custom-column {{
            padding: 20px;
            background-color: #EBF5FB;  /* Azul pastel muy claro para el fondo de columnas */
            border-radius: 10px;
            color: #3C3C3C;  /* Gris oscuro suave para el texto */
            margin-bottom: 25px;
            font-size: 1rem;
        }}
        .custom-offset {{
            margin-top: 30px;  /* Margen superior para alinear verticalmente */
        }}
        .custom-footer {{
            color: #3C3C3C;  /* Gris oscuro suave para el pie de página */
            text-align: center;
            font-style: italic;
            margin-top: 30px;
            font-size: 1.0rem;
        }}
        h6 {{
            font-size: 0.9rem;
            color: #6E6E6E;  /* Gris medio para subtítulos */
        }}
    </style>
"""