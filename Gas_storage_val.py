import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple


class GasStorageContract:

    def __init__(self, price_data_path: str):
        self.prices_df = pd.read_csv(price_data_path)
        self.prices_df['Dates'] = pd.to_datetime(self.prices_df['Dates'])
        self.prices_df['Prices'] = self.prices_df['Prices'].astype(float)

    def get_price(self, date: str) -> float:
        target_date = pd.to_datetime(date)
        idx = (self.prices_df['Dates'] - target_date).abs().idxmin()
        return self.prices_df.loc[idx, 'Prices']

    def calculate_storage_volume(self,
                                 injection_schedule: List[Dict],
                                 withdrawal_schedule: List[Dict],
                                 injection_rate: float,
                                 withdrawal_rate: float,
                                 max_volume: float) -> Tuple[bool, str, List[Dict]]:
        timeline = []

        for inj in injection_schedule:
            timeline.append({
                'date': pd.to_datetime(inj['date']),
                'type': 'injection',
                'volume': inj['volume']
            })

        for wth in withdrawal_schedule:
            timeline.append({
                'date': pd.to_datetime(wth['date']),
                'type': 'withdrawal',
                'volume': wth['volume']
            })

        timeline.sort(key=lambda x: x['date'])

        current_volume = 0
        volume_history = []

        for event in timeline:
            if event['type'] == 'injection':
                if event['volume'] > injection_rate:
                    return False, f"Injection rate exceeded on {event['date'].date()}: {event['volume']} > {injection_rate}", []

                current_volume += event['volume']

                if current_volume > max_volume:
                    return False, f"Max volume exceeded on {event['date'].date()}: {current_volume} > {max_volume}", []

            else:
                if event['volume'] > withdrawal_rate:
                    return False, f"Withdrawal rate exceeded on {event['date'].date()}: {event['volume']} > {withdrawal_rate}", []

                current_volume -= event['volume']

                if current_volume < 0:
                    return False, f"Negative volume on {event['date'].date()}: {current_volume}", []

            volume_history.append({
                'date': event['date'],
                'type': event['type'],
                'volume_change': event['volume'] if event['type'] == 'injection' else -event['volume'],
                'total_volume': current_volume
            })

        return True, "All constraints satisfied", volume_history

    def calculate_storage_costs(self,
                               injection_schedule: List[Dict],
                               withdrawal_schedule: List[Dict],
                               storage_cost_per_unit: float) -> float:
        all_events = []

        for inj in injection_schedule:
            all_events.append((pd.to_datetime(inj['date']), 'injection', inj['volume']))

        for wth in withdrawal_schedule:
            all_events.append((pd.to_datetime(wth['date']), 'withdrawal', wth['volume']))

        all_events.sort(key=lambda x: x[0])

        total_cost = 0
        current_volume = 0

        for i, (date, event_type, volume) in enumerate(all_events):
            if event_type == 'injection':
                current_volume += volume
            else:
                current_volume -= volume

            if i < len(all_events) - 1:
                next_date = all_events[i + 1][0]
                days = (next_date - date).days
            else:
                days = 1

            total_cost += current_volume * days * storage_cost_per_unit

        return total_cost

    def price_contract(self,
                      injection_schedule: List[Dict],
                      withdrawal_schedule: List[Dict],
                      injection_rate: float,
                      withdrawal_rate: float,
                      max_volume: float,
                      storage_cost_per_unit: float,
                      verbose: bool = True) -> Dict:

        is_valid, message, volume_timeline = self.calculate_storage_volume(
            injection_schedule, withdrawal_schedule,
            injection_rate, withdrawal_rate, max_volume
        )

        if not is_valid:
            if verbose:
                print(f"INVALID CONTRACT: {message}")
            return {
                'contract_value': None,
                'total_injection_cost': None,
                'total_withdrawal_revenue': None,
                'total_storage_cost': None,
                'is_valid': False,
                'error_message': message,
                'volume_timeline': []
            }

        total_injection_cost = 0
        injection_details = []

        for inj in injection_schedule:
            price = self.get_price(inj['date'])
            cost = inj['volume'] * price
            total_injection_cost += cost
            injection_details.append({
                'date': inj['date'],
                'volume': inj['volume'],
                'price': price,
                'cost': cost
            })

        total_withdrawal_revenue = 0
        withdrawal_details = []

        for wth in withdrawal_schedule:
            price = self.get_price(wth['date'])
            revenue = wth['volume'] * price
            total_withdrawal_revenue += revenue
            withdrawal_details.append({
                'date': wth['date'],
                'volume': wth['volume'],
                'price': price,
                'revenue': revenue
            })

        total_storage_cost = self.calculate_storage_costs(
            injection_schedule, withdrawal_schedule, storage_cost_per_unit
        )

        contract_value = total_withdrawal_revenue - total_injection_cost - total_storage_cost

        if verbose:
            print("=" * 80)
            print("NATURAL GAS STORAGE CONTRACT VALUATION")
            print("=" * 80)
            print("\nCONTRACT SUMMARY")
            print("-" * 80)

            print("\nINJECTION SCHEDULE (Buying Gas - Cash OUT):")
            print(f"{'Date':<15} {'Volume':<15} {'Price':<15} {'Cost':<15}")
            print("-" * 60)
            for inj in injection_details:
                print(f"{inj['date']:<15} {inj['volume']:<15,.0f} ${inj['price']:<14,.2f} ${inj['cost']:<14,.2f}")
            print("-" * 60)
            print(f"{'TOTAL INJECTION COST:':<45} ${total_injection_cost:>14,.2f}")

            print("\nWITHDRAWAL SCHEDULE (Selling Gas - Cash IN):")
            print(f"{'Date':<15} {'Volume':<15} {'Price':<15} {'Revenue':<15}")
            print("-" * 60)
            for wth in withdrawal_details:
                print(f"{wth['date']:<15} {wth['volume']:<15,.0f} ${wth['price']:<14,.2f} ${wth['revenue']:<14,.2f}")
            print("-" * 60)
            print(f"{'TOTAL WITHDRAWAL REVENUE:':<45} ${total_withdrawal_revenue:>14,.2f}")

            print("\nSTORAGE COSTS:")
            print(f"Storage rate: ${storage_cost_per_unit}/MMBtu/day")
            print(f"{'TOTAL STORAGE COST:':<45} ${total_storage_cost:>14,.2f}")

            print("\n" + "=" * 80)
            print("CONTRACT VALUE CALCULATION")
            print("=" * 80)
            print(f"Withdrawal Revenue:        ${total_withdrawal_revenue:>14,.2f}")
            print(f"Injection Cost:          - ${total_injection_cost:>14,.2f}")
            print(f"Storage Cost:            - ${total_storage_cost:>14,.2f}")
            print("-" * 80)
            print(f"NET CONTRACT VALUE:        ${contract_value:>14,.2f}")
            print("=" * 80)

            if contract_value > 0:
                print(f"\nThis contract is PROFITABLE by ${contract_value:,.2f}")
            elif contract_value < 0:
                print(f"\nThis contract LOSES ${abs(contract_value):,.2f}")
            else:
                print("\nThis contract BREAKS EVEN")

            print("\nVOLUME TIMELINE:")
            print(f"{'Date':<15} {'Event':<15} {'Volume Change':<20} {'Total in Storage':<20}")
            print("-" * 70)
            for event in volume_timeline:
                print(f"{event['date'].date()!s:<15} {event['type'].title():<15} "
                      f"{event['volume_change']:>18,.0f} {event['total_volume']:>18,.0f}")

        return {
            'contract_value': contract_value,
            'total_injection_cost': total_injection_cost,
            'total_withdrawal_revenue': total_withdrawal_revenue,
            'total_storage_cost': total_storage_cost,
            'is_valid': is_valid,
            'injection_details': injection_details,
            'withdrawal_details': withdrawal_details,
            'volume_timeline': volume_timeline
        }


if __name__ == "__main__":

    pricer = GasStorageContract('Nat_Gas.csv')

    injection_schedule = [
        {'date': '2021-05-31', 'volume': 100000}
    ]

    withdrawal_schedule = [
        {'date': '2021-12-31', 'volume': 100000}
    ]

    result = pricer.price_contract(
        injection_schedule=injection_schedule,
        withdrawal_schedule=withdrawal_schedule,
        injection_rate=150000,
        withdrawal_rate=150000,
        max_volume=500000,
        storage_cost_per_unit=0.005,
        verbose=True
    )

    print(f"\nFinal Contract Value: ${result['contract_value']:,.2f}")
