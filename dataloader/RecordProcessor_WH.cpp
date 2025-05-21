#include "RecordProcessor.h"

using namespace System;
using namespace System::Collections::Generic;
using namespace Npgsql;

int RecordProcessor::ProcessWHRecord(array<Byte>^ record) {
	try {
		NpgsqlCommand^ command = gcnew NpgsqlCommand(nullptr, connection);
		
		// Extract common data
		String^ dataType = ByteSubstring(record, 2, 1);
		DateTime^ creationDate = DateTime::ParseExact(ByteSubstring(record, 3, 8), "yyyyMMdd", nullptr);
		DateTime^ kaisaiDate = DateTime::ParseExact(ByteSubstring(record, 11, 8), "yyyyMMdd", nullptr);
		String^ keibajoCode = ByteSubstring(record, 19, 2);
		Int16 kaisaiKai = Int16::Parse(ByteSubstring(record, 21, 2));
		Int16 kaisaiNichime = Int16::Parse(ByteSubstring(record, 23, 2));
		Int16 kyosoBango = Int16::Parse(ByteSubstring(record, 25, 2));
		
		// Check if any records already exist for this race
		command->CommandText = "SELECT creation_date FROM wh WHERE "
			"kaisai_date = @kaisai_date AND "
			"keibajo_code = @keibajo_code AND "
			"kaisai_kai = @kaisai_kai AND "
			"kaisai_nichime = @kaisai_nichime AND "
			"kyoso_bango = @kyoso_bango "
			"ORDER BY creation_date DESC LIMIT 1";
		
		command->Parameters->AddWithValue("@kaisai_date", kaisaiDate);
		command->Parameters->AddWithValue("@keibajo_code", keibajoCode);
		command->Parameters->AddWithValue("@kaisai_kai", kaisaiKai);
		command->Parameters->AddWithValue("@kaisai_nichime", kaisaiNichime);
		command->Parameters->AddWithValue("@kyoso_bango", kyosoBango);
		
		Object^ existingCreationDateObj = command->ExecuteScalar();
		
		// If records exist and they're newer than our data, skip
		if (existingCreationDateObj != nullptr) {
			DateTime^ existingCreationDate = Convert::ToDateTime(existingCreationDateObj);
			if (existingCreationDate->CompareTo(creationDate) > 0)
				return PROCESS_SUCCESS;
				
			// If our data is newer, delete all existing records for this race
			command->CommandText = "DELETE FROM wh WHERE "
				"kaisai_date = @kaisai_date AND "
				"keibajo_code = @keibajo_code AND "
				"kaisai_kai = @kaisai_kai AND "
				"kaisai_nichime = @kaisai_nichime AND "
				"kyoso_bango = @kyoso_bango";
			command->ExecuteNonQuery();
		}
		
		// バッチ処理のためのリストを作成
		List<String^>^ batchValues = gcnew List<String^>();
		
		// Process each horse
		for (int i = 0; i < 18; i++) {
			Int16 umaban;
			
			try {
				umaban = Int16::Parse(ByteSubstring(record, 35 + (45 * i), 2));
			}
			catch (...) {
				// Skip if we can't parse a valid umaban (likely empty/padding)
				continue;
			}
			
			// Skip if umaban is 0 or missing data
			if (umaban <= 0) 
				continue;
			
			// Handle bataiju
			String^ bataijuValue = "NULL";
			String^ bataijuStr = ByteSubstring(record, 35 + 38 + (45 * i), 3);
			if (bataijuStr != "   " && bataijuStr != "000" && bataijuStr != "999") {
				try {
					Int16 bataiju = Int16::Parse(bataijuStr);
					bataijuValue = bataiju.ToString();
				}
				catch (...) {
					// Leave as NULL
				}
			}
			
			// Handle zogensa
			String^ zogensaValue = "NULL";
			String^ zogensaStr = ByteSubstring(record, 35 + 41 + (45 * i), 3);
			if (zogensaStr != "    " && zogensaStr != " 999") {
				try {
					Int16 zogensa = Int16::Parse(zogensaStr);
					zogensaValue = zogensa.ToString();
				}
				catch (...) {
					// Leave as NULL
				}
			}
			
			// Format the values string for this horse
			String^ valueStr = String::Format("('{0}', '{1}', '{2}', '{3}', '{4}', '{5}', '{6}', {7}, {8}, {9})",
				dataType, 
				creationDate->ToString("yyyy-MM-dd"), 
				kaisaiDate->ToString("yyyy-MM-dd"),
				keibajoCode,
				kaisaiKai,
				kaisaiNichime,
				kyosoBango,
				umaban,
				bataijuValue,
				zogensaValue);
				
			batchValues->Add(valueStr);
		}
		
		// 一括挿入を実行
		if (batchValues->Count > 0) {
			String^ valuesSql = String::Join(", ", batchValues);
			command->CommandText = "INSERT INTO wh (data_type, creation_date, "
				"kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, "
				"umaban, bataiju, zogensa) VALUES " + valuesSql;
			command->Parameters->Clear();
			command->ExecuteNonQuery();
		}
		
		return PROCESS_SUCCESS;
	}
	catch (Exception^ ex) {
		Console::WriteLine();
		Console::WriteLine("Failed to process WH record: {0}", ex->Message);
		return PROCESS_ERROR;
	}
}