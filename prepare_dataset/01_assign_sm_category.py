import pandas as pd
import os
import sys
from timeit import default_timer as timer
from datetime import timedelta
import click

def domains_collect(x,y):
    y = set(map(str.strip, y.split(',')))
    
    if type(x) is str:
        x = set(map(str.strip, x.split(',')))
        y = y.union(x)
        
    return list(y)

@click.command()
@click.option('--source_data_folder', default="../data/raw", help='Raw data folder')
@click.option('--service_map_path', default="../data/Service_MAP.csv", help='Path to service map')
@click.option('--save_name', default="../data/preprocessed/flows_SM.pkl", help='Output file path')
def load_and_save_data(source_data_folder, service_map_path, save_name): 
    # Load data from CSV files that are in the folders splitted by days
    DAYS = os.listdir(source_data_folder)
    DAYS.sort()
    print('Day folders:', DAYS)  # Print the names of day folders

    # Load data to one big dataframe, save as .pkl, print loading and saving times
    start = timer()
    dfs = []

    for day in DAYS:
        if day.startswith("20"):
            print('-' * 50)
            print('date:', day)
            data_dir = os.path.join(source_data_folder, day)
            csvs = os.listdir(data_dir)
            csvs.sort()

            for csv in csvs:
                if csv.startswith("flows"):
                    file = os.path.join(data_dir, csv)
                    print(file)
                    df_temp = pd.read_csv(file)
                    df_temp['datetime'] = csv.split('.')[1]
                    df_temp["orig_source_file"] = file
                    df_temp.index.name = 'orig_line_index'
                    df_temp.reset_index(inplace = True)
                    dfs.append(df_temp)

    df = pd.concat(dfs)
    end = timer()
  
    print('-' * 50)
    print('time to load csvs:', timedelta(seconds=end - start))

    # Mapping categories from the service map
    df_sm = pd.read_csv(service_map_path)  # Path to file with service map
    
    
    ##################################################################################
    
    # preprare the service map for merge
    
    df_sm.drop(columns = ['Tag', 'Netify', 'Netify Link', 'Endpoints Description', 'Netify Domains', 'Netify Domains (cached)'],inplace = True)
    
    df_sm['Domains_collected'] = df_sm.apply(lambda x: domains_collect(x['Manual Domains'],x['Merged Domains']), axis=1)
    df_sm.drop(columns = ['Manual Domains', 'Merged Domains'],inplace = True)
       
    # explode 
    df_sme = df_sm.explode('Domains_collected')
    # numerical target 
    df_sme.index.rename('SM_category', inplace=True)
    df_sme.reset_index(inplace=True)

    # hloubka domény a zda zastupuje několik domén
    df_sme['Domain_alias'] = df_sme['Domains_collected'].apply(lambda x: 1 if '*' in x else 0)
    # odříznout *, schválně necháme . aby se rozlišilo, kdy má adresa přesný tvar a kdy jde o alias
    df_sme['Domains_collected'] = df_sme['Domains_collected'].apply(lambda x: x.strip('*'))
    
    
    
    #####################################################################################
   # Checking that every TLS_SNI is mapped to some domain from service map

    # Collecting domains *.domain.com
    aliases = df_sme[df_sme['Domain_alias'] == 1]['Domains_collected']
    not_aliases = df_sme[df_sme['Domain_alias'] == 0]['Domains_collected']
    TLS_SNI_MAP = pd.DataFrame(df['TLS_SNI'].unique(), columns=['TLS_SNI'])

    # Collected domains from the service map
    TLS_SNI_MAP['Domains_collected'] = ''

    for val in aliases:
        TLS_SNI_MAP.loc[TLS_SNI_MAP['TLS_SNI'].str.endswith(val), 'Domains_collected'] = val

    for val in aliases:
        TLS_SNI_MAP.loc[TLS_SNI_MAP['TLS_SNI'].str.match(val[1:]), 'Domains_collected'] = val

    for val in not_aliases:
        TLS_SNI_MAP.loc[TLS_SNI_MAP['TLS_SNI'].str.match(val), 'Domains_collected'] = val

    # Checking that everything is mapped   
    unmapped_tls_sni = TLS_SNI_MAP[TLS_SNI_MAP['Domains_collected'] == '']
    if not unmapped_tls_sni.empty:
        print(unmapped_tls_sni, 'are not mapped to domains from the service map')

    #################################################################################################

    # Merge
    TLS_SNI_MAP_SC = TLS_SNI_MAP.merge(df_sme, how ='left', left_on ='Domains_collected', right_on ='Domains_collected')
    
    # Mapping SM_CATEGOTY to df
    df_mapped = df.merge(TLS_SNI_MAP_SC, how='left', left_on='TLS_SNI', right_on='TLS_SNI')

    # Put targets to the beginning
    sm_targets = ['Domains_collected', 'SM_category', 'Service', 'Service Group', 'Service Category',
                  'Domain_alias']
    df_mapped = df_mapped[sm_targets + [col for col in df_mapped.columns if col not in sm_targets]]

    
    
    # drop everything unneeded
    columns_to_drop = ['TLS_SNI','Domain_alias']
        
    # Save
    df_mapped.drop(columns_to_drop,inplace = True, axis =1)
    
    print('saving dataframe with mapped categories from the service map')
    df_mapped.to_pickle(save_name)

if __name__ == "__main__":
    load_and_save_data()