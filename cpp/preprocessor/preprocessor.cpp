#include <iostream>
#include <fstream>
#include "jvdata.h"

using namespace std::literals;

int main(int argc, char *argv[])
{
	if (argc == 1)
	{
		std::cerr << "Usage: " << argv[0] << " [testcommon|testdata]" << std::endl;
		return 1;
	}

	if (strcmp(argv[1], "testdata") == 0)
	{

		try
		{
			// specify the path to the data file (data/jvd.dat)
			std::string dataFilePath = "data/jvd.dat";

			// check if the file exists
			std::ifstream file(dataFilePath);
			if (!file)
			{
				std::cerr << "Error: File not found: " << dataFilePath << std::endl;
				return 1;
			}

			// check the file size
			file.seekg(0, std::ios::end);
			std::streamsize fileSize = file.tellg();
			file.seekg(0, std::ios::beg);
			std::cout << "File size: " << fileSize << " bytes" << std::endl;

			// read the file line by line
			int recordCount = 0;
			std::string line;

			while (std::getline(file, line))
			{
				// add 1 to the record count
				recordCount++;

				try
				{
					// Create the appropriate Record object based on line content
					std::unique_ptr<JVData::Record> record = JVData::RecordFactory::createRecord(line);

					// Type-specific processing if needed
					if (record->getRecordType() == "TK")
					{
						// Common processing
						std::cout << "Record " << recordCount << ":" << std::endl;
						std::cout << "  Record Type: " << record->getRecordType() << std::endl;
						std::cout << "  Data Type: " << record->getDataType() << std::endl;
						std::cout << "  Creation Date: " << record->getCreationDate() << std::endl;

						// TKRecord specific processing
						auto tkRecord = dynamic_cast<const JVData::TKRecord *>(record.get());
						if (tkRecord)
						{
							std::cout << "  Year: " << tkRecord->getYear() << std::endl;
							std::cout << "  Month/Day: " << tkRecord->getMonthDay() << std::endl;
							std::cout << "  Keibajo Code: " << tkRecord->getKeibajoCode() << std::endl;
							std::cout << "  Kaisai Kai: " << tkRecord->getKaisaiKai() << std::endl;
							std::cout << "  Kaisai Nichime: " << tkRecord->getKaisaiNichime() << std::endl;
							std::cout << "  Kyoso Bango: " << tkRecord->getKyosoBango() << std::endl;
						}
					}
				}
				catch (const std::exception &e)
				{
					std::cerr << "Error processing record " << recordCount << ": " << e.what() << std::endl;
					return 1;
				}
			}

			file.close();

			std::cout << "Total records: " << recordCount << std::endl;
			return 0;
		}

		catch (const std::exception &e)
		{
			std::cerr << "Error: " << e.what() << std::endl;
			return 1;
		}
	}

	else
	{
		std::cerr << "Error: Invalid argument provided.";
		return 1;
	}
}
