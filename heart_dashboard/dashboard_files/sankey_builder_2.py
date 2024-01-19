# The main graph generator developed
import plotly.graph_objects as go
import pandas as pd


def code_mapping(df, df_columns=None, cull = False, cull_val = 20):
    '''
    Args:
    :param df: the dataframe being key matched
    :param df_columns: a list of the specific columns being grouped,
    sorted, and matched to a numeric key
    :param cull: cuts down large datasets to a smaller data set
    this only works well will a bi-level sankey diagram because the data flow is
    disjoint in multilevel diagrams
    :param cull_val removes all count values less than this value
    :returns:
    :return_df: the sorted dataframe containing the values as their
    numeric key and the value columns
    :labels: a list of the specific string labels that represent each number key
    value assigned to the elements of the dataframe
    '''

    if df_columns is None: # Initializes list and dataframe
        df_columns = []
    stacked = pd.DataFrame()
    for i in range(len(df_columns) - 1): # Loops through selected rows and compiles them into one condensed df
        grouped_df = df.groupby([df_columns[i], df_columns[i + 1]]).size().reset_index(name='Count')
        grouped_df = grouped_df.rename(columns={df_columns[i]: 'Source'})
        grouped_df = grouped_df.rename(columns={df_columns[i + 1]: 'Target'})
        stacked = pd.concat([stacked, grouped_df])
        if cull == True: # Cuts down graph size
            stacked = stacked[stacked['Count'] > cull_val]


    # Makes long list of labels from all the columns
    labels = list(set(list(stacked['Source']) + list(stacked['Target'])))

    # generate n integers for n labels
    codes = list(range(len(labels)))

    # create a map from label to code
    lc_map = dict(zip(labels, codes))
    # substitute names for codes in the dataframe

    stacked = stacked.replace({'Source': lc_map, 'Target': lc_map})
    return stacked, labels

    # generate n integers for n labels
    codes = list(range(len(labels)))
    # create a map from label to code
    lc_map = dict(zip(labels, codes))

    # substitute names for codes in the dataframe
    return_df = pd.DataFrame({'Sources': [lc_map], 'Targets': [lc_map], 'Values': [values]})
    return return_df, labels # Returns the created objects


def make_sankey(df, df_columns, title='', cull = False, cull_val = 20):
    """
    :Purpose: This function takes a dataframe and selected columns and generates a sankey diagram
    :Args:
    :param df: the dataframe that one wants to make a sankey diagram from
    :param df_columns: a list of columns that want to be included in the diagram
    :param title: the title of the plot
    :param cull: asks user if they want to filter out less common data
    :param cull_val asks user what number of count values to go up to before item is included in diagram
    :return:
    :display: opens the generated graph in a browser

    """
    # Calls the function above to create a grouped dataframe of the columns passed in
    df_new, labels = code_mapping(df, df_columns, cull, cull_val)

    # Generates a dictionary of the sources targets and values assigned to the sankey graph
    link = {'source': df_new['Source'], 'target': df_new['Target'], 'value': df_new['Count'],
            'line': {'color': '#8a0a0a', 'width': 1}}

    # Generates a dictionary of the labels on the graph items so the graph is readable
    node = {'label': labels, 'pad': 5, 'thickness': 50,
            'line': {'color': 'black', 'width': 2}}

    sk = go.Sankey(link=link, node=node) # Builds the diagram
    fig = go.Figure(sk) # Builds the graph from the diagram

    fig.update_layout( # Styles and labels the graph
        hovermode='x',
        title=str(title),
        font=dict(size=13, color='black'),
        paper_bgcolor='#ffffff'
    )

    return fig # Launches the figure


