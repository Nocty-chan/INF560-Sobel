#include <iostream>
#include <fstream>
#include <cstring>
#include <stdlib.h>
using namespace std;

const int MAX_CHARS_PER_LINE = 512;
const int MAX_TOKENS_PER_LINE = 20;
const char* const DELIMITER = " ";

int main(int argc, char *argv[])
{
  string filepath;
  if (argc == 2) {
    filepath = argv[1];
  } else {
    cout << "Please specify filepath." << endl;
    return -1;
  }

  // create a file-reading object
  ifstream fin;
  fin.open(filepath.c_str()); // open a file
  if (!fin.good())
    return 1; // exit if file not found

  string currentGifName ="";
  int nbImages = 0;
  double grayFilterCost = 0, blurFilterCost = 0, sobelFilterCost = 0, totalCost = 0;

  // read each line of the file
  while (!fin.eof())
  {
    // read an entire line into memory
    char buf[MAX_CHARS_PER_LINE];
    // array to store memory addresses of the tokens in buf
    const char* token[MAX_TOKENS_PER_LINE] = {}; // initialize to 0
    fin.getline(buf, MAX_CHARS_PER_LINE);
    // parse the line into blank-delimited tokens

    int n= 0;
    // parse the line
    token[0] = strtok(buf, DELIMITER); // first token
    if (token[0]) // zero if line is blank
    {
      for (n = 1; n < MAX_TOKENS_PER_LINE; n++)
      {
        token[n] = strtok(0, DELIMITER); // subsequent tokens
        if (!token[n]) break; // no more tokens
      }

      string firstWord(token[0]);
      if (currentGifName.compare("") == 0 && firstWord.compare("GIF") == 0) {
          currentGifName = token[4];
          nbImages = atoi(token[6]);
      } else if (grayFilterCost == 0 && firstWord.compare("Gray") == 0){
          grayFilterCost = atof(token[5]);
      } else if (blurFilterCost == 0 && firstWord.compare("Blur") == 0) {
          blurFilterCost = atof(token[5]);
      } else if (sobelFilterCost == 0 && firstWord.compare("Sobel") == 0) {
          sobelFilterCost = atof(token[5]);
      } else if (totalCost == 0 && firstWord.compare("Filtering") == 0) {
         totalCost = atof(token[3]);
      }

      if (currentGifName != "" && nbImages > 0 && grayFilterCost > 0 && blurFilterCost > 0 && sobelFilterCost > 0 && totalCost > 0) {
        cout << "Name: " << currentGifName << endl;
        cout << "Number of images: " << nbImages << endl;
        cout << "Gray filter: " << grayFilterCost << endl;
        cout << "Blur filter: " << blurFilterCost << endl;
        cout << "Sobel filter: " << sobelFilterCost << endl;
        cout << "Total Cost: " << totalCost << endl;
        totalCost = 0;
        sobelFilterCost = 0;
        blurFilterCost = 0;
        grayFilterCost = 0;
        nbImages = 0;
        currentGifName = "";
      }
    }
  }
}
