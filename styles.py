
class Theme:
    # Dark Theme Colors
    DARK = {
        "BG_MAIN": "#121212",
        "BG_SUB": "#1e1e1e",
        "TEXT_MAIN": "#ffffff",
        "TEXT_SUB": "#e0e0e0",
        "PRIMARY": "#1a237e",
        "PRIMARY_LIGHT": "#2962ff",
        "ACCENT": "#00c853",
        "BORDER": "#333333",
        "INPUT_BG": "#2d2d2d"
    }

    # Light Theme Colors
    LIGHT = {
        "BG_MAIN": "#f5f5f5",
        "BG_SUB": "#ffffff",
        "TEXT_MAIN": "#000000",
        "TEXT_SUB": "#424242",
        "PRIMARY": "#3f51b5",
        "PRIMARY_LIGHT": "#5c6bc0",
        "ACCENT": "#00e676",
        "BORDER": "#e0e0e0",
        "INPUT_BG": "#ffffff"
    }

    CURRENT_THEME = "Light"

    @staticmethod
    def get_colors():
        return Theme.DARK if Theme.CURRENT_THEME == "Dark" else Theme.LIGHT

    @staticmethod
    def get_stylesheet():
        c = Theme.get_colors()
        return f"""
        QMainWindow {{
            background-color: {c["BG_MAIN"]};
            color: {c["TEXT_MAIN"]};
        }}
        QWidget {{
            background-color: {c["BG_MAIN"]};
            color: {c["TEXT_MAIN"]};
            font-family: "Segoe UI", Arial, sans-serif;
            font-size: 14px;
        }}
        
        /* Top Bar */
        QFrame#TopBar {{
            background-color: {c["BG_SUB"]};
            border-bottom: 1px solid {c["BORDER"]};
        }}
        QLabel#LogoTitle {{
            font-size: 28px;
            font-weight: 800;
            color: {c["TEXT_MAIN"]};
            padding: 12px;
            background-color: transparent;
            font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
        }}
        
        /* Side Navigation */
        QFrame#SideNav {{
            background-color: {c["BG_SUB"]};
            border-right: 1px solid {c["BORDER"]};
            min-width: 90px;
            max-width: 90px;
        }}
        QToolButton[class="NavButton"] {{
            background-color: transparent;
            border: none;
            color: {c["TEXT_SUB"]};
            padding: 5px;
            font-size: 13px;
            border-radius: 8px;
        }}
        QToolButton[class="NavButton"]:hover {{
            background-color: {c["PRIMARY"]};
            color: #ffffff;
        }}
        QToolButton[class="NavButton"]:checked {{
            background-color: {c["PRIMARY_LIGHT"]};
            color: #ffffff;
        }}
        
        /* Side Panel & Content */
        QWidget#SidePanelContent {{
            background-color: {c["BG_SUB"]};
        }}
        QGroupBox {{
            border: 1px solid {c["BORDER"]};
            border-radius: 8px;
            margin-top: 20px;
            font-weight: bold;
            color: {c["TEXT_MAIN"]};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 3px;
            color: {c["PRIMARY_LIGHT"]};
        }}
        
        /* Controls */
        QPushButton[class="ActionButton"] {{
            background-color: {c["PRIMARY"]};
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 600;
        }}
        QPushButton[class="ActionButton"]:hover {{
            background-color: {c["PRIMARY_LIGHT"]};
        }}

        /* Status Label (Rounded Pill) */
        QLabel#StatusLabel {{
            background-color: {c["BG_MAIN"]};
            border: 1px solid {c["BORDER"]};
            border-radius: 12px;
            padding: 4px 12px;
            font-weight: bold;
        }}
        QPushButton[class="SecondaryButton"] {{
            background-color: transparent;
            color: {c["TEXT_MAIN"]};
            border: 1px solid {c["BORDER"]};
            border-radius: 6px;
            padding: 6px 12px;
        }}
        QPushButton[class="SecondaryButton"]:hover {{
            background-color: {c["BG_MAIN"]};
            border-color: {c["PRIMARY"]};
            color: {c["PRIMARY"]};
        }}
        
        /* Inputs & Lists */
        QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
            background-color: {c["INPUT_BG"]};
            border: 1px solid {c["BORDER"]};
            border-radius: 4px;
            padding: 4px;
            color: {c["TEXT_MAIN"]};
        }}
        /* Hide native spinbox arrows (unreliable on some Windows/DPI setups);
           explicit +/- QPushButton added by _num_row() is used instead. */
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            width: 0;
            border: none;
        }}
        QListWidget {{
            background-color: {c["INPUT_BG"]};
            border: 1px solid {c["BORDER"]};
            border-radius: 4px;
            color: {c["TEXT_MAIN"]};
        }}
        QTextEdit {{
            background-color: {c["BG_SUB"]};
            color: {c["TEXT_SUB"]};
            border-top: 1px solid {c["BORDER"]};
        }}
        
        /* System log 文档框完整边框 */
        QTextEdit#ConsoleLog {{
            background-color: {c["BG_SUB"]};
            color: {c["TEXT_SUB"]};
            border: 1px solid {c["BORDER"]};
            border-radius: 4px;
        }}
        
        /* Tabs */
        QTabWidget::pane {{
            border: 1px solid {c["BORDER"]};
            background: {c["BG_MAIN"]};
        }}
        QTabBar::tab {{
            background: {c["BG_SUB"]};
            color: {c["TEXT_SUB"]};
            padding: 8px 12px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}
        QTabBar::tab:selected {{
            background: {c["PRIMARY_LIGHT"]};
            color: #ffffff;
        }}
        
        /* Dock Widget */
        QDockWidget::title {{
            background: {c["PRIMARY"]};
            color: #ffffff;
            text-align: left; 
            padding-left: 5px;
        }}
        """
