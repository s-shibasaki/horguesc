#pragma once
ref class DataLoader
{
private:
	AxJVDTLabLib::AxJVLink m_jvlink;
	bool Initialize();

public:
	DataLoader();
	bool Execute();
};

