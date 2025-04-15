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

	JVLinkClass^ jvlink = gcnew JVLinkClass();
	jvlink->JVInit("UNKNOWN");
	std::cout << "JVLink initialized." << std::endl;

	std::vector<Record> records;
	for (int i = 0; i < 100; ++i) {
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