import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

def df_to_pdf(df, pdf_filename=None, folder_path='results/pdf'):
    """
    Save a pandas DataFrame to a PDF file, by default in the 'results/pdf' folder.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to save to PDF
    pdf_filename : str, default None
        Name of the PDF file (without path). If None, uses 'dataframe.pdf'
    folder_path : str, default 'results/pdf'
        Directory where to save the PDF file
    
    Returns:
    --------
    str
        The full path to the saved PDF file
    """
    # Set default filename if not provided
    if pdf_filename is None:
        pdf_filename = 'dataframe.pdf'
    
    # Create full path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    full_path = os.path.join(folder_path, pdf_filename)
    
    # Create a PDF document
    with PdfPages(full_path) as pdf:
        # Create a figure and axis
        # widen the page as columns are added so the table stays readable
        fig, ax = plt.subplots(figsize=(max(12, len(df.columns) * 1.3), 4))
        
        # Hide the axes
        ax.axis('off')
        
        # Create a table and add it to the axis
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            loc='center',
            cellLoc='center'
        )
        
        # Adjust table style
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Add the figure to PDF
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"DataFrame successfully saved to {full_path}")
    return full_path
