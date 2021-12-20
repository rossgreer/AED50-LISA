import pandas as pd

### Functions to make pedestrian-only dataset and phase signal datasets.


def gen_ped_cycle_set():
	"""Writes a CSV file with only pedestrian and cyclist entries (vehicles removed)"""
	df = pd.read_csv('Competition_Data/Lidar/collection_three.csv')
	print(len(df))
	pedestrian_set = df[(df['Label']==2) | (df['Label']==3)]
	print(len(pedestrian_set))
	pedestrian_set.to_csv('Competition_Data/Lidar/pedestrians_only_three.csv')

non_ped_messages = ['Data Gap Alarm', 'Unit Flash Status', 'Unit Alarm Status 1', 'Alarm Group Status', 'Special Func Output Off  (Func #1)', 'Special Func Output Off  (Func #2)',
                    'Special Func Output Off  (Func #3)', 'Special Func Output Off  (Func #4)', 'Special Func Output Off  (Func #5)', 'Special Func Output Off  (Func #6)', 
                    'Special Func Output Off  (Func #7)', 'Special Func Output Off  (Func #8)','Coord Pattern  (Pattern #6)', 'Coord Cycle Length = 80s', 'Coord Offset = 54s',
                    'Programmed Split 1 Length = 13s', 'Programmed Split 2 Length = 32s', 'Programmed Split 3 Length = 0s', 'Programmed Split 4 Length = 35s', 
                    'Programmed Split 5 Length = 0s', 'Programmed Split 6 Length = 0s', 'Programmed Split 7 Length = 0s', 'Programmed Split 8 Length = 0s', 
                    'Programmed Split 9 Length = 0s', 'Programmed Split 10 Length = 0s', 'Programmed Split 11 Length = 0s', 'Programmed Split 12 Length = 0s', 
                    'Programmed Split 13 Length = 0s', 'Programmed Split 14 Length = 0s', 'Programmed Split 15 Length = 0s', 'Programmed Split 16 Length = 0s', 
                    'Detector On  (Veh Det #10)', 'Phase End Yellow Cl  (Phase 2)', 'Phase Begin Red Cl  (Phase 2)', 'Overlap Begin Red Cl  (Overlap  (Overlap A)', 
                    'Detector Off  (Veh Det #2)', 'Detector On  (Veh Det #2)', 'Detector Off  (Veh Det #1)', 'Phase Call Dropped  (Phase 1)',  'Phase End Red Cl  (Phase 2)',
                    'Phase Begin Green  (Phase 4)', 'Overlap Off  (Overlap  (Overlap A)', 'Barrier Term  (Preceding Phase 1)', 'Detector Off  (Veh Det #10)',  
                    'Detector Off  (Veh Det #15)', 'Phase Min Complete  (Phase 4)', 'Detector Off  (Veh Det #4)', 'Phase Call Dropped  (Phase 4)', 'Phase Green Term  (Phase 4)',
                    'Phase Begin Yellow Cl  (Phase 4)', 'Phase Gap Out  (Phase 4)', 'Detector On  (Veh Det #12)', 'Detector Off  (Veh Det #12)', 'Detector On  (Veh Det #11)', 
                    'Phase End Yellow Cl  (Phase 4)', 'Phase Begin Red Cl  (Phase 4)', 'Detector Off  (Veh Det #11)',  'Phase End Red Cl  (Phase 4)',  'Phase Begin Green  (Phase 2)',
                    'Bike Call Off  (Phase 2)', 'Bike Call On  (Phase 2)', 'Overlap Begin Green  (Overlap  (Overlap A)', 'Ped Detector On  (Ped Det #4)', 'Phase Check  (Phase 4)',
                    'Ped Call Registered  (Ped 4)', 'Ped Detector Off  (Ped Det #4)', 'Phase Omit On  (Phase 4)', 'Phase Min Complete  (Phase 2)', 'Detector On  (Veh Det #14)', 
                    'Phase Gap Out  (Phase 2)', 'Detector Off  (Veh Det #14)', 'Detector On  (Veh Det #4)', 'Phase Call Registered  (Phase 4)', 'Phase Omit On  (Phase 1)', 
                    'Detector On  (Veh Det #13)', 'Detector On  (Veh Det #15)', 'Detector Off  (Veh Det #13)', 'Coord Cycle State', 'Phase Hold Released  (Phase 2)', 
                    'Detector On  (Veh Det #9)', 'Detector Off  (Veh Det #9)', 'Detector On  (Veh Det #16)', 'Detector Off  (Veh Det #16)', 'Detector On  (Veh Det #1)',
                    'Coord Phase Yield Pt  (Phase 4)', 'Phase Green Term  (Phase 2)', 'Phase Begin Yellow Cl  (Phase 2)', 'Phase Check  (Phase 1)', 'Phase Omit Off  (Phase 2)',
                    'Phase Omit Off  (Phase 4)', 'Phase Call Registered  (Phase 1)', 'Overlap Begin Yellow Cl  (Overlap  (Overlap A)', 'Coord Phase Yield Pt  (Phase 1)', 
                    'Coord Phase Yield Pt  (Phase 3)', 'Phase Check  (Phase 2)', 'Phase Omit Off  (Phase 1)', 'Phase Force Off  (Phase 4)', 'Phase On  (Phase 1)',
                    'Phase Begin Green  (Phase 1)', 'Phase Min Complete  (Phase 1)', 'Phase Green Term  (Phase 1)', 'Phase Begin Yellow Cl  (Phase 1)', 
                    'Phase Force Off  (Phase 1)', 'Phase End Yellow Cl  (Phase 1)', 'Phase Begin Red Cl  (Phase 1)', 'Phase Inactive  (Phase 1)', 'Phase End Red Cl  (Phase 1)',
                    'Phase Gap Out  (Phase 1)',  'Ped Omit On  (Ped 1)', 'Phase Omit On  (Phase 2)','Ped Omit Off  (Ped 1)']

ped4_related_messages = ['Ped Begin Walk  (Ped 4)', 'Ped Begin Clearance  (Ped 4)', 'Ped Begin Don’t Walk  (Ped 4)']
ped2_related_messages = ['Ped Begin Walk  (Ped 2)', 'Ped Begin Clearance  (Ped 2)','Ped Begin Don’t Walk  (Ped 2)']

phase2_related_messages = ['Phase Inactive  (Phase 2)', 'Phase Hold Active  (Phase 2)','Phase On  (Phase 2)']
phase4_related_messages = ['Phase On  (Phase 4)','Phase Inactive  (Phase 4)','Phase Hold Active  (Phase 4)']

omit2_related_messages = ['Ped Omit On  (Ped 2)', 'Ped Omit Off  (Ped 2)']
omit4_related_messages = ['Ped Omit On  (Ped 4)', 'Ped Omit Off  (Ped 4)']


def gen_ped2_commands():
	""" Writes a CSV file for each of the phase information sets, containing information about the pedestrian crossing only."""
	two_messages = []
	four_messages = []
	two_phases = []
	four_phases = []
	four_omits = []
	# for each row of the SPM file
	df = pd.read_csv('Competition_Data/SPM/updated_complete_spm.csv')

	for index, row in df.iterrows():
		if row['Message'] in ped2_related_messages: #or row['Message'] in phase2_related_messages or row['Message'] in omit2_related_messages:
			two_messages += [[row['Message'], row['Timestamp']]]
		elif row['Message'] in ped4_related_messages: #or row['Message'] in phase4_related_messages or row['Message'] in omit4_related_messages:
			four_messages += [[row['Message'], row['Timestamp']]]

	df2 = pd.DataFrame(two_messages, columns=['Message', 'Timestamp'])
	df4 = pd.DataFrame(four_messages, columns=['Message', 'Timestamp'])
	df2.to_csv('Competition_Data/SPM/signal_two_part_one.csv')
	df4.to_csv('Competition_Data/SPM/signal_four_part_one.csv')

#gen_ped2_commands()
#gen_ped_cycle_set()


