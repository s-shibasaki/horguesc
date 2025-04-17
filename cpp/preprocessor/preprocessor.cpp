#include <iostream>
#include <fstream>
#include "jvdata.h"

using namespace std::literals;

int main(int argc, char *argv[])
{
	if (argc == 1)
	{
		std::cerr << "ERROR: No arguments provided."s;
		return 1;
	}

	if (strcmp(argv[1], "testcommon") == 0)
	{

		try
		{
			std::string str;

			str = "20230405"s;
			std::cout << "Testing YMD: " << str << std::endl;
			std::cout << JVData::YMD(str).toString() << std::endl;

			str = "20230405060708"s;
			std::cout << "Testing RACE_ID: " << str << std::endl;
			std::cout << JVData::RACE_ID(str).toString() << std::endl;

			str = "2023040506070809"s;
			std::cout << "Testing RACE_ID: " << str << std::endl;
			std::cout << JVData::RACE_ID(str).toString() << std::endl;

			str = "001002003004005006"s;
			std::cout << "Testing CHAKUKAISU_INFO: " << str << std::endl;
			std::cout << JVData::CHAKUKAISU_INFO(str).toString() << std::endl;

			str = "000100020003000400050006"s;
			std::cout << "Testing CHAKUKAISU_INFO: " << str << std::endl;
			std::cout << JVData::CHAKUKAISU_INFO(str).toString() << std::endl;

			str = "202301234567890123456789000001000002000003000004000005000006"s;
			std::cout << "Testing SEI_RUIKEI_INFO: " << str << std::endl;
			std::cout << JVData::SEI_RUIKEI_INFO(str).toString() << std::endl;

			str = "20230123456789012345678901234567890123456789"s +
				  "000001000002000003000004000005000006"s +
				  "000007000008000009000010000011000012"s +
				  "000013000014000015000016000017000018"s +
				  "000019000020000021000022000023000024"s +
				  "000025000026000027000028000029000030"s +
				  "000031000032000033000034000035000036"s +
				  "000037000038000039000040000041000042"s +
				  "000043000044000045000046000047000048"s +
				  "000049000050000051000052000053000054"s +
				  "000055000056000057000058000059000060"s +
				  "000061000062000063000064000065000066"s +
				  "000067000068000069000070000071000072"s +
				  "000073000074000075000076000077000078"s +
				  "000079000080000081000082000083000084"s +
				  "000085000086000087000088000089000090"s +
				  "000091000092000093000094000095000096"s +
				  "000097000098000099000100000101000102"s +
				  "000103000104000105000106000107000108"s +
				  "000109000110000111000112000113000114"s +
				  "000115000116000117000118000119000120"s +
				  "000121000122000123000124000125000126"s +
				  "000127000128000129000130000131000132"s +
				  "000133000134000135000136000137000138"s +
				  "000139000140000141000142000143000144"s +
				  "000145000146000147000148000149000150"s +
				  "000151000152000153000154000155000156"s +
				  "000157000158000159000160000161000162"s +
				  "000163000164000165000166000167000168"s;
			std::cout << "Testing HON_ZEN_RUIKEISEI_INFO: " << str << std::endl;
			std::cout << JVData::HON_ZEN_RUIKEISEI_INFO(str).toString() << std::endl;

			std::cout << "All tests completed successfully."s;
			return 0;
		}

		catch (const std::exception &e)
		{
			std::cerr << "ERROR: " << e.what() << std::endl;
			return 1;
		}
	}

	else if (strcmp(argv[1], "testdata") == 0)
	{

		try
		{
			// specify the path to the data file (data/jvd.dat)
			std::string dataFilePath = "data/jvd.dat";

			// check if the file exists
			std::ifstream file(dataFilePath);
			if (!file)
			{
				std::cerr << "ERROR: File not found: " << dataFilePath << std::endl;
				return 1;
			}
			std::cout << "File found: " << dataFilePath << std::endl;

			// check the file size
			file.seekg(0, std::ios::end);
			std::streamsize fileSize = file.tellg();
			file.seekg(0, std::ios::beg);
			std::cout << "File size: " << fileSize << " bytes" << std::endl;

			int topN = 1;

			// reset counters (for all records, for per-record-spec (TK, RA, SE, ...))
			int recordCount = 0;
			int recordCountTK = 0;
			int recordCountRA = 0;

			// read every line of the file
			std::string line;
			while (std::getline(file, line))
			{
				// add 1 to the record count
				recordCount++;

				// check if the head 2 characters are "TK"
				if (line.substr(0, 2) == "TK")
				{
					if (++recordCountTK > topN)
						continue;

					JVData::JV_TK_TOKUUMA record(line);
					std::cout << record.toString() << std::endl;
				}

				else if (line.substr(0, 2) == "RA")
				{
					if (++recordCountRA > topN)
						continue;

					JVData::JV_RA_RACE record(line);
					std::cout << record.toString() << std::endl;
				}
			}

			file.close();

			std::cout << "All lines read successfully."s;
			std::cout << "Total records: " << recordCount << std::endl;
			std::cout << "Total TK records: " << recordCountTK << std::endl;
			return 0;
		}

		catch (const std::exception &e)
		{
			std::cerr << "ERROR: " << e.what() << std::endl;
			return 1;
		}
	}

	else
	{
		std::cerr << "ERROR: Invalid argument provided."s;
		return 1;
	}
}
