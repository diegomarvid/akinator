# %%
from akinator import SimplifiedAkinator

akinator = SimplifiedAkinator('data/personajes.csv', debug=True)
akinator.print_tree()
akinator.play()
# %%
