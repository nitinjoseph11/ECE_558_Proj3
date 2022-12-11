from tkinter import *
from PIL import ImageTk, Image
root = Tk()

root.geometry("700x350")

canv = Canvas(root, width=300, height=400, bg='white')
canv.pack(pady=20)
#canv.grid(row=2, column=3)

img = ImageTk.PhotoImage(Image.open("greyGirl.png"))  # PIL solution
my_image=canv.create_image(80, 80, anchor=NW, image=img)

def left(e):
   x = -20
   y = 0
   canv.move(my_image, x, y)

def right(e):
   x = 20
   y = 0
   canv.move(my_image, x, y)

def up(e):
   x = 0
   y = -20
   canv.move(my_image, x, y)

def down(e):
   x = 0
   y = 20
   canv.move(my_image, x, y)

# Bind the move function
root.bind("<Left>", left)
root.bind("<Right>", right)
root.bind("<Up>", up)
root.bind("<Down>", down)

root.mainloop()


#mainloop()