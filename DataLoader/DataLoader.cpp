#include "pch.h"

struct Record {
	int id;
	float value;
};

int main(int argc, char** argv)
{

	CLI::App app{ "Data Loader Application" };
	std::string output_path;
	app.add_option("--output", output_path, "Output file path")->required();
	try {
		app.parse(argc, argv);
	}
	catch (const CLI::ParseError& e) {
		return app.exit(e);
	}

	int returncode;

	JVLink^ jvlink = gcnew JVLinkClass();
	returncode = jvlink->JVInit("UNKNOWN");
	if (returncode != 0) {
		Console::Error->WriteLine("JVLink initialization failed with error code: {0}", returncode);
		return 1;
	}
	Console::WriteLine("JVLink initialized successfully.");

	int readcount = 0;
	int downloadcount = 0;
	System::String^ lastfiletimestamp = "";

	returncode = jvlink->JVOpen("RACE", "20180101000000-20191231000000", 4, readcount, downloadcount, lastfiletimestamp);
	if (returncode != 0) {
		Console::Error->WriteLine("JVOpen failed with error code: {0}", returncode);
		return 1;
	}
	Console::WriteLine("JVOpen succeeded.");
	Console::WriteLine("Read count: {0}", readcount);
	Console::WriteLine("Download count: {0}", downloadcount);
	Console::WriteLine("Last file timestamp: {0}", lastfiletimestamp);

	Console::Write("Downloading data: {0} / {1}", 0, downloadcount);
	while (true) {
		returncode = jvlink->JVStatus();
		if (returncode >= 0) {
			Console::Write("\rDownloading data: {0} / {1}", returncode, downloadcount);
			if (returncode == downloadcount) {
				Console::WriteLine("\nAll data downloaded.");
				break;
			}
		}
		else
		{
			Console::Error->WriteLine("\nJVStatus failed with error code: {0}", returncode);
			return 1;
		}
	}
	
	Object^ buff = gcnew array<Byte>(110000);
	String^ filename = "";

	array<Byte>^ byteArray;
	Text::Encoding^ shiftJis = Text::Encoding::GetEncoding("shift_jis");
	String^ content;
	int done = 0;

	Console::Write("Reading data: {0} / {1}", 0, readcount);
	while (true) {
		returncode = jvlink->JVGets(buff, 110000, filename);
		if (returncode > 0) {
			// データを読み込んだ場合
			byteArray = safe_cast<array<Byte>^>(buff);
			content = shiftJis->GetString(byteArray, 0, returncode);
		}
		else if (returncode >= -1) {
			// ファイル終端に達した場合
			Console::Write("\rReading data: {0} / {1}", ++done, readcount);
			if (returncode == 0) {
				Console::WriteLine("\nAll data read.");
				break;
			}
		}
		else {
			Console::Error->WriteLine("\nJVRead failed with error code: {0}", returncode);
			return 1;
		}
	}



	std::vector<Record> records;
	for (int i = 0; i < 3; ++i) {
		Record record;
		record.id = i;
		record.value = static_cast<float>(i) * 1.1f;
		records.push_back(record);
	}

	int version = 1;
	size_t record_count = records.size();

	std::ofstream output_file(output_path, std::ios::binary);
	if (output_file.is_open()) {
		output_file.write(reinterpret_cast<const char*>(&version), sizeof(int));
		output_file.write(reinterpret_cast<const char*>(&record_count), sizeof(size_t));
		output_file.write(reinterpret_cast<const char*>(records.data()), record_count * sizeof(Record));
		output_file.close();
		std::cout << "Data written to " << output_path << std::endl;
	}
	else {
		std::cerr << "Failed to open output file: " << output_path << std::endl;
		return 1;
	}

	return 0;
}