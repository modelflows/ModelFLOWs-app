import matplotlib.animation as animation
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from IPython.display import HTML

def animated_plot(path0, Tensor, vel, Title):
    '''
    Function that creates an animated contourf plot
    
    Args:
        path0 - path where the graph is to be saved to later be loaded on the streamlit app
        Tensor - data file
        vel - velocity variable: 0 for x velocity; 1 for y velocity
        Title - Title for the graph (i.e. original data, reconstructed data...)
    '''

    frames = Tensor.shape[-1]

    fig, ax = plt.subplots()

    def animate(i):
        ax.clear()
        ax.contourf(Tensor[vel, :, :, i]) 
        ax.set_title(Title)

    interval = 2     
    anim = animation.FuncAnimation(fig, animate, frames = frames, interval = interval*1e+2, blit = False)

    plt.show()

    with open(f"{path0}/animation.html","w") as f:
        print(anim.to_html5_video(), file = f)

    HtmlFile = open(f"{path0}/animation.html", "r")

    source_code = HtmlFile.read()

    components.html(source_code, height = 900, width=900)
