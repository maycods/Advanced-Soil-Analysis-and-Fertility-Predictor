import utils
import pandas as pd
import matplotlib.pyplot as plt
import squarify
import io

class StatisticsCOVID19:
    """
    Class for plotting COVID-19 statistics.

    Attributes:
    - df: The DataFrame containing COVID-19 statistics.

    Methods:
    - plot_total_cases_and_positive_tests(chosen_attribute): Plot the distribution of total cases or positive tests by zone.
    - plot_total_cases_and_positive_tests_treemap(chosen_attribute): Plot a treemap of the distribution of total cases or positive tests by zone.
    - weekly_plot(chosen_zone, chosen_year, chosen_month, chosen_attribute): Plot the weekly evolution of total cases or positive tests or total tests for a specific zone during a chosen month and year.
    - monthly_plot(chosen_zone, chosen_year, chosen_attribute): Plot the monthly evolution of total cases or positive tests or total tests for a specific zone during a chosen year.
    - annual_plot(chosen_zone, chosen_attribute): Plot the annual evolution of total cases or positive tests or total tests for a specific zone.
    - stacked_bar_plot(): Plot a stacked bar graph of case count by zone and year.
    - pop_tests_plot(): Plot population versus test count.
    - plot_top_zones_impacted(n): Plot the top n zones most impacted by the Coronavirus.
    - plot_time_period_data(chosen_time_period, chosen_attribute): Plot the total of chosen attribute for a given time period.
    """

    def __init__(self, df):
        self.df = pd.DataFrame(df)
        self.df['Year'] = pd.to_datetime(self.df['Start date']).dt.year
        self.df['Month'] = pd.to_datetime(self.df['Start date']).dt.month

    def plot_total_cases_and_positive_tests(self, chosen_attribute):
        totals = self.df.groupby('zcta')[[chosen_attribute]].sum().reset_index()
        bar_width = 0.5
        index = totals.index
        plt.bar(index, totals[chosen_attribute], bar_width, label=chosen_attribute)
        plt.xlabel('Zones')
        plt.ylabel('Count')
        plt.title(f'Distribution du nombre total de {chosen_attribute} par zones')

    def plot_total_cases_and_positive_tests_treemap(self, chosen_attribute):
        totals = self.df.groupby('zcta')[[chosen_attribute]].sum().reset_index()

        totals['value_normalized'] = totals[chosen_attribute] / totals[chosen_attribute].sum()

        totals = totals.sort_values(by=chosen_attribute, ascending=False)

        colors = plt.cm.tab10(range(len(totals)))

        fig, ax = plt.subplots(figsize=(10, 6))
        squarify.plot(
            sizes=totals['value_normalized'],
            label=totals['zcta'],
            color=colors,
            alpha=0.7,
            ax=ax 
        )

        plt.title(f'Distribution du nombre de {chosen_attribute} par zone')
        plt.axis('off') 
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close(fig)

        # Save the buffer to the specified filename
        with open("plots\\total_cases_and_positive_tests_treemap_plot.png", 'wb') as f:
            f.write(buffer.getvalue())

        return "plots\\total_cases_and_positive_tests_treemap_plot.png" 

    def weekly_plot(self, chosen_zone, chosen_year, chosen_month, chosen_attribute):

        zone_df = self.df[self.df['zcta'] == chosen_zone]

        hebdo_df = zone_df[(zone_df['Month'] == chosen_month) & (zone_df['Year'] == chosen_year)]

        plt.plot(hebdo_df['Start date'], hebdo_df[chosen_attribute], label=chosen_attribute)
        plt.title(f'L\'évolution hebdomadaire du total de {chosen_attribute} pour la zone {chosen_zone} pendant le {chosen_month} ème mois de l\'année {chosen_year}')
        plt.xlabel('Dates')
        plt.ylabel('Count')

    def monthly_plot(self, chosen_zone, chosen_year, chosen_attribute):

        zone_df = self.df[self.df['zcta'] == chosen_zone]

        monthly_df = zone_df[zone_df['Year'] == chosen_year]
        month_df = monthly_df.groupby('Month')[[chosen_attribute]].sum().reset_index()
        plt.plot(month_df['Month'], month_df[chosen_attribute], label=chosen_attribute)
        plt.title(f'L\'évolution mensuelle du total de {chosen_attribute} pour la zone {chosen_zone} pendant l\'année {chosen_year}')
        plt.xlabel('Months')
        plt.ylabel('Count') 
        
    def annual_plot(self, chosen_zone, chosen_attribute):

        zone_df = self.df[self.df['zcta'] == chosen_zone]

        annual_df = zone_df.groupby('Year')[[chosen_attribute]].sum().reset_index()

        plt.plot(annual_df['Year'], annual_df[chosen_attribute], label=chosen_attribute)
        plt.title(f'L\'évolution annuelle du total de {chosen_attribute} pour la zone {chosen_zone}')
        plt.xlabel('Years')
        plt.ylabel('Count') 
        plt.xticks(annual_df['Year'])


    def stacked_bar_plot(self):
        self.df = self.df.sort_values(by=['Year'])  # Sort the DataFrame by 'Year'

        grouped_data = self.df.groupby(['Year', 'zcta'])['case count'].sum().unstack()

        years = self.df['Year'].unique()
        zone_columns = grouped_data.columns

        bottom_values = None

        for zone in zone_columns:
            values = grouped_data[zone].reindex(years, fill_value=0).values
            if bottom_values is None:
                plt.bar(years, values, label=zone)
                bottom_values = values
            else:
                plt.bar(years, values, label=zone, bottom=bottom_values)
                bottom_values += values

        plt.title('Stacked Bar Graph of Case Count by Zone and Year')
        plt.xlabel('Year')
        plt.ylabel('Case Count')
        plt.legend(title='Zone', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(years) 


    def pop_tests_plot(self):
        data = self.df[['population', 'test count']]
        grouped_data = data.groupby('population').sum().reset_index()

        plt.plot(grouped_data['population'], grouped_data['test count'])
        plt.title('Population vs Test Count')
        plt.xlabel('Population')
        plt.ylabel('Test count')

    def plot_top_zones_impacted(self, n):
        grouped_data = self.df.groupby('zcta')['case count'].sum().sort_values(ascending=True).head(n)
        bar_width = 0.1

        grouped_data.plot.barh(figsize=(12, 6), width=bar_width)
        plt.title(f'Top {n} Zones les plus impactées par le Coronavirus')
        plt.xlabel('Nombre de Cas')

    def plot_time_period_data(self, chosen_time_period, chosen_attribute):
        selected_data = self.df[(self.df['time_period'] == chosen_time_period)]
        grouped_data = selected_data.groupby('zcta')[[chosen_attribute]].sum().reset_index()

        bar_width = 0.2
        index = grouped_data.index

        plt.bar(index - bar_width, grouped_data[chosen_attribute], width=bar_width, label=chosen_attribute)
        plt.xticks(index - bar_width, grouped_data['zcta'])
        plt.xlabel('Zone (zcta)')
        plt.ylabel('Count')
        plt.title(f'Total of {chosen_attribute} for Time Period {chosen_time_period}')
        plt.legend()