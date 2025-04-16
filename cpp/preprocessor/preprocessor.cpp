#include "pch.h"
#include <iostream>
#include "jvdata.h"

using namespace System;

int main(array<System::String ^> ^args)
{
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
