#include "cudaMain.h"
#include "main.h"
#include "wind.h"

IMPLEMENT_APP(MyApp)

bool MyApp::OnInit()
{
    Simple *simple = new Simple(wxT("Simple"));
    simple->Show(true);

    return cudaMain(argc, argv);
}

/*
int main(int argc, char **argv)
{

    return cudaMain(argc, argv);
}
*/
