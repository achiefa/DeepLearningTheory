from typing import List
import matplotlib.pyplot as plt

from utils import XGRID, fk_ev_map

def plot_eigvals(eigvals, title='', **kwargs):
  '''
  Plot the eigenvalues
  '''

  fig, ax = plt.subplots(**kwargs)

  ax.plot(eigvals, label='Eigenvalue', color='royalblue', linestyle='', marker='o')

  # Customize the grid
  ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

  # Set title
  ax.set_title(title, fontsize=16, pad=15)

  # Customize ticks
  ax.tick_params(axis='both', which='major', labelsize=12)
  ax.tick_params(axis='both', which='minor', labelsize=10)

  # Set labels
  ax.set_xlabel('Index', fontsize=20)
  ax.set_ylabel(r'$\lambda_k$', fontsize=20)

  # Set scale
  ax.set_xscale('symlog')
  ax.set_yscale('symlog')

  # Add a legend
  ax.legend(fontsize=15)
  
  return fig, ax

def plot_eigvals_and_eigvecs(eigvals, eigvecs, title_vals='', title_vecs='', show_vecs=[0], **kwargs):
  figs = []
  axes = []
  fig, ax = plot_eigvals(eigvals, titke=title_vals, **kwargs)
  figs.append(fig)
  axes.append(ax)

  # Plot for eigenvecs



def plot_PDFs(pdfs, flat_to_pdf=False, label='', **kwargs):
  '''
  Plot the 9 flavours that take part in the regression. The values of
  the PDFs can be given either in the from (50, 9) or the flatten
  version (450,). For the latter case, the flag `flat_to_pdf` must
  be set to True.
  '''
  pdfs_tensor = pdfs
  if flat_to_pdf:
    pdfs_tensor = pdfs_tensor.reshape((9,50))

  fig, axes = plt.subplots(3, 3,  **kwargs)
  
  for i, ax in enumerate(axes.flat):

    ax.plot(XGRID, pdfs_tensor[i,:], label=label)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(fk_ev_map[i], fontsize=20)
    ax.set_xscale('log')
    ax.set_xlim(1e-5, 1.0)
    ax.legend()

  #axes[-1, -1].axis('off')
  plt.tight_layout
  return fig, axes