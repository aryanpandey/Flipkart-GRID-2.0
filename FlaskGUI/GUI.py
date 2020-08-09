from tkinter import *
import tkinter.filedialog as filedialog

class smallGUI():

    def __init__(self):
        self.check = False
        self.first = Tk()  
        self.first.title(string = "Third Degree Burn")
        self.first.geometry("400x300")
        self.button1 = Button(self.first, text = "Run Single Audio FIle", command = self.singleinput, width = 30, height = 2).place(x = 90, y = 90)
        self.button2 = Button(self.first, text = "Run Multiple Audio files", command = self.multipleinput, width = 30, height = 2).place(x = 90, y = 190)
        self.first.mainloop() 

    def singleinput(self):
        self.check = True
        self.first.destroy()
        self.tk = Tk()  
        self.tk.title(string = "Third Degree Burn")
        self.tk.geometry("400x300")
        self.button3 = Button(self.tk, text = "Select Input File", command = self.inputfile, width = 30, height = 2).place(x = 90, y = 90)
        self.button4 = Button(self.tk, text = "Select Directory to Save Outputs", command = self.outputdir, width = 30, height = 2).place(x = 90, y = 190)
        self.tk.mainloop() 

    def multipleinput(self):
        self.first.destroy()
        self.tk = Tk()  
        self.tk.title(string = "Third Degree Burn")
        self.tk.geometry("400x300")
        self.button3 = Button(self.tk, text = "Select Directory with Input Files", command = self.inputdir, width = 30, height = 2).place(x = 90, y = 90)
        self.button4 = Button(self.tk, text = "Select Directory to Save Outputs", command = self.outputdir, width = 30, height = 2).place(x = 90, y = 190)
        self.tk.mainloop() 

    def inputfile(self):
        self.inputfile = filedialog.askopenfilename(title = "Audio file")

    def inputdir(self):
        self.inputdirpath = filedialog.askdirectory(title="Open Input folder")

    def outputdir(self):
        self.outputdirpath = filedialog.askdirectory(title="Open Output folder")
        self.tk.destroy()
