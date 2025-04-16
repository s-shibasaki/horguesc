#include "pch.h"
#include <iostream>
#include "jvdata.h"

using namespace System;
using namespace JVData;
using namespace std::literals;

int main(array<System::String^>^ args) {

	if (args->Length == 0) {
		Console::WriteLine("ERROR: No arguments provided.");
		return 1;
	};

	if (args[0] == "testjvdata") {

		RACE_ID race_id;
		CHAKUKAISU_INFO chakukaisu_info;
		SEI_RUIKEI_INFO sei_ruikei_info;
		HON_ZEN_RUIKEISEI_INFO hon_zen_ruikei_info;

		Console::WriteLine("\nTesting RACE_ID with RaceNum:");
		race_id = RACE_ID("2023040506070809");
		Console::WriteLine("Year: {0}", gcnew String(race_id.Year.c_str()));
		Console::WriteLine("MonthDay: {0}", gcnew String(race_id.MonthDay.c_str()));
		Console::WriteLine("JyoCD: {0}", gcnew String(race_id.JyoCD.c_str()));
		Console::WriteLine("Kaiji: {0}", gcnew String(race_id.Kaiji.c_str()));
		Console::WriteLine("Nichiji: {0}", gcnew String(race_id.Nichiji.c_str()));
		Console::WriteLine("RaceNum: {0}", gcnew String(race_id.RaceNum.c_str()));

		Console::WriteLine("\nTesting RACE_ID without RaceNum:");
		race_id = RACE_ID("20230405060708");
		Console::WriteLine("Year: {0}", gcnew String(race_id.Year.c_str()));
		Console::WriteLine("MonthDay: {0}", gcnew String(race_id.MonthDay.c_str()));
		Console::WriteLine("JyoCD: {0}", gcnew String(race_id.JyoCD.c_str()));
		Console::WriteLine("Kaiji: {0}", gcnew String(race_id.Kaiji.c_str()));
		Console::WriteLine("Nichiji: {0}", gcnew String(race_id.Nichiji.c_str()));
		Console::WriteLine("RaceNum: {0}", gcnew String(race_id.RaceNum.c_str()));


		Console::WriteLine("\nTesting RECORD_ID (3 bytes):");
		chakukaisu_info = CHAKUKAISU_INFO("001002003004005006");
		for (int i = 0; i < 6; ++i) {
			Console::WriteLine("Chakukaisu[{0}]: {1}", i, gcnew String(chakukaisu_info.Chakukaisu[i].c_str()));
		}

		Console::WriteLine("\nTesting RECORD_ID (4 bytes):");
		chakukaisu_info = CHAKUKAISU_INFO("000100020003000400050006");
		for (int i = 0; i < 6; ++i) {
			Console::WriteLine("Chakukaisu[{0}]: {1}", i, gcnew String(chakukaisu_info.Chakukaisu[i].c_str()));
		}

		Console::WriteLine("\nTesting SEI_RUIKEI_INFO:");
		sei_ruikei_info = SEI_RUIKEI_INFO("202301234567890123456789000001000002000003000004000005000006");
		Console::WriteLine("SetYear: {0}", gcnew String(sei_ruikei_info.SetYear.c_str()));
		Console::WriteLine("HonSyokinTotal: {0}", gcnew String(sei_ruikei_info.HonSyokinTotal.c_str()));
		Console::WriteLine("FukaSyokin: {0}", gcnew String(sei_ruikei_info.FukaSyokin.c_str()));
		for (int i = 0; i < 6; ++i) {
			Console::WriteLine("ChakuKaisu[{0}]: {1}", i, gcnew String(sei_ruikei_info.ChakuKaisu.Chakukaisu[i].c_str()));
		}

		Console::WriteLine("\nTesting SEI_RUIKEI_INFO (default constructor):");
		sei_ruikei_info = SEI_RUIKEI_INFO();
		Console::WriteLine("SetYear: {0}", gcnew String(sei_ruikei_info.SetYear.c_str()));
		Console::WriteLine("HonSyokinTotal: {0}", gcnew String(sei_ruikei_info.HonSyokinTotal.c_str()));
		Console::WriteLine("FukaSyokin: {0}", gcnew String(sei_ruikei_info.FukaSyokin.c_str()));
		for (int i = 0; i < 6; ++i) {
			Console::WriteLine("ChakuKaisu[{0}]: {1}", i, gcnew String(sei_ruikei_info.ChakuKaisu.Chakukaisu[i].c_str()));
		}

		Console::WriteLine("\nTesting HON_ZEN_RUIKEISEI_INFO:");
		hon_zen_ruikei_info = HON_ZEN_RUIKEISEI_INFO(
			"20230123456789012345678901234567890123456789"s +
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
			"000163000164000165000166000167000168"s
		);
		Console::WriteLine("SetYear: {0}", gcnew String(hon_zen_ruikei_info.SetYear.c_str()));
		Console::WriteLine("HonSyokinHeichi: {0}", gcnew String(hon_zen_ruikei_info.HonSyokinHeichi.c_str()));
		Console::WriteLine("HonSyokinSyogai: {0}", gcnew String(hon_zen_ruikei_info.HonSyokinSyogai.c_str()));
		Console::WriteLine("FukaSyokinHeichi: {0}", gcnew String(hon_zen_ruikei_info.FukaSyokinHeichi.c_str()));
		Console::WriteLine("FukaSyokinSyogai: {0}", gcnew String(hon_zen_ruikei_info.FukaSyokinSyogai.c_str()));
		for (int i = 0; i < 6; ++i) {
			Console::WriteLine("ChakuKaisuHeichi[{0}]: {1}", i, gcnew String(hon_zen_ruikei_info.ChakuKaisuHeichi.Chakukaisu[i].c_str()));
		}
		for (int i = 0; i < 6; ++i) {
			Console::WriteLine("ChakuKaisuSyogai[{0}]: {1}", i, gcnew String(hon_zen_ruikei_info.ChakuKaisuSyogai.Chakukaisu[i].c_str()));
		}
		for (int i = 0; i < 20; ++i) {
			for (int j = 0; j < 6; ++j) {
				Console::WriteLine("ChakuKaisuJyo[{0}][{1}]: {2}", i, j, gcnew String(hon_zen_ruikei_info.ChakuKaisuJyo[i].Chakukaisu[j].c_str()));
			}
		}
		for (int i = 0; i < 6; ++i) {
			for (int j = 0; j < 6; ++j) {
				Console::WriteLine("ChakuKaisuKyori[{0}][{1}]: {2}", i, j, gcnew String(hon_zen_ruikei_info.ChakuKaisuKyori[i].Chakukaisu[j].c_str()));
			}
		}

		
		Console::WriteLine("\nTest completed.");
		return 0;
	}

	else if (args[0] == "testloaddata") {
		// specify file path
		String^ filePath = "data/jvd.dat";

		Console::WriteLine("File: " + filePath);

		// check if file exists
		if (!IO::File::Exists(filePath)) {
			Console::WriteLine("ERROR: File not found: " + filePath);
			return 1;
		}

		// Get and display file size
		IO::FileInfo^ fileInfo = gcnew IO::FileInfo(filePath);
		double fileSizeMB = fileInfo->Length / (1024.0 * 1024.0);
		Console::WriteLine("File size: {0:F2} MB", fileSizeMB);

		// Create StreamReader
		IO::StreamReader^ reader = gcnew IO::StreamReader(filePath);
		Console::WriteLine("Starting file read...\n");

		int lineCount = 0;
		int lineCountUnknown = 0;
		String^ line;

		while (line = reader->ReadLine()) {
			lineCount++;
			if (line->Substring(0, 2) == "TK") {
				array<Byte>^ bytes = System::Text::Encoding::GetEncoding("shift-jis")->GetBytes(line);
				pin_ptr<Byte> pinnedBytes = &bytes[0];

				// TODO
			}
			else {
				lineCountUnknown++;
			}
		}

		reader->Close();
		Console::WriteLine("\nFile reading completed.");
		Console::WriteLine("Unknown {0} of {1} lines.", lineCountUnknown, lineCount);

		return 0;
	}

	else {
		Console::WriteLine("ERROR: Unknown command.");
		return 1;
	}
}
