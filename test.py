# %%
from akinator_terminal_gui import create_akinator_terminal_gui

akinator_app = create_akinator_terminal_gui("data/personajes.csv", debug=True)
akinator_app.play()

# %%
