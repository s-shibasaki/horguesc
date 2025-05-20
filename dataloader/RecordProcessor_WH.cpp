#include "RecordProcessor.h"

using namespace System;
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
		
		// Clear parameters for reuse
		command->Parameters->Clear();
		
		// Prepare the insert command
		command->CommandText = "INSERT INTO wh (data_type, creation_date, "
			"kaisai_date, keibajo_code, kaisai_kai, kaisai_nichime, kyoso_bango, "
			"umaban, bataiju, zogensa) "
			"VALUES (@data_type, @creation_date, @kaisai_date, @keibajo_code, "
			"@kaisai_kai, @kaisai_nichime, @kyoso_bango, @umaban, @bataiju, @zogensa)";
			
		// Add common parameters
		command->Parameters->AddWithValue("@data_type", dataType);
		command->Parameters->AddWithValue("@creation_date", creationDate);
		command->Parameters->AddWithValue("@kaisai_date", kaisaiDate);
		command->Parameters->AddWithValue("@keibajo_code", keibajoCode);
		command->Parameters->AddWithValue("@kaisai_kai", kaisaiKai);
		command->Parameters->AddWithValue("@kaisai_nichime", kaisaiNichime);
		command->Parameters->AddWithValue("@kyoso_bango", kyosoBango);
		
		// Add parameter objects (will update values for each insert)
		NpgsqlParameter^ umabanParam = gcnew NpgsqlParameter("@umaban", NpgsqlTypes::NpgsqlDbType::Smallint);
		NpgsqlParameter^ bataijuParam = gcnew NpgsqlParameter("@bataiju", NpgsqlTypes::NpgsqlDbType::Smallint);
		NpgsqlParameter^ zogensaParam = gcnew NpgsqlParameter("@zogensa", NpgsqlTypes::NpgsqlDbType::Smallint);
		
		command->Parameters->Add(umabanParam);
		command->Parameters->Add(bataijuParam);
		command->Parameters->Add(zogensaParam);
		
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
			
			umabanParam->Value = umaban;
			
			// Handle bataiju
			String^ bataijuStr = ByteSubstring(record, 35 + 38 + (45 * i), 3);
			if (bataijuStr != "   " && bataijuStr != "000" && bataijuStr != "999") {
				try {
					bataijuParam->Value = Int16::Parse(bataijuStr);
				}
				catch (...) {
					bataijuParam->Value = DBNull::Value;
				}
			}
			else {
				bataijuParam->Value = DBNull::Value;
			}
			
			// Handle zogensa
			String^ zogensaStr = ByteSubstring(record, 35 + 41 + (45 * i), 3);
			if (zogensaStr != "    " && zogensaStr != " 999") {
				try {
					zogensaParam->Value = Int16::Parse(zogensaStr);
				}
				catch (...) {
					zogensaParam->Value = DBNull::Value;
				}
			}
			else {
				zogensaParam->Value = DBNull::Value;
			}
			
			// Insert the record
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