from clipper.clipper import Clipper
from juwazaku.posefitter import PoseFitter


def main_clipper():
	# video_path = 'C:\\Users\\escac\\Projects\\Academics\\jwk\\res\\SAR202210D2T1FIN.webm'
	# csv_path = 'C:\\Users\\escac\\Projects\\Academics\\jwk\\res\\SAR202210D2T1FIN.csv'
	video_path = 'C:\\Users\\escac\\Projects\\Academics\\jwk\\res\\LIS2104D1T1.webm'
	csv_path = 'C:\\Users\\escac\\Projects\\Academics\\jwk\\res\\LIS2104D1T1.csv'
	savedir = 'C:\\Users\\escac\\Projects\\Academics\\jwk\\res\\out'
	name = 'LIS2104D1T1'

	with Clipper(video_path, savedir, name) as clipper:
		clipper.export_from_csv(csv_path, sim=False)


def main_posefitter():
	# video_path = 'C:\\Users\\escac\\Projects\\Academics\\jwk\\res\\out\\LIS2104D1T1-8220-8340.mp4'
	video_path = 'C:\\Users\\escac\\Projects\\Academics\\jwk\\res\\out\\SAR202210D2T1FIN-9810-9960.mp4'

	with PoseFitter(video_path, "posefitter", 'C:\\Users\\escac\\Projects\\Academics\\jwk\\res\\out') as posefitter:
		posefitter.export(debug=True)


if __name__ == '__main__':
	main_posefitter()
