/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2020 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    solidFoam

Group
    grpHeatTransferSolvers

Description
    Solver for energy transport and thermodynamics on a solid.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "dynamicFvMesh.H"
#include "solidThermo.H"
#include "radiationModel.H"
#include "fvOptions.H"
#include "simpleControl.H"
#include "pimpleControl.H"
#include "dummyCourantNo.H"
#include "solidRegionDiffNo.H"
#include "coordinateSystem.H"
#include "omp.h"
#include "gpu.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Solver for energy transport and thermodynamics on a solid"
    );

    #define NO_CONTROL
    #include "postProcess.H"

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createDynamicFvMesh.H"
    #include "createFields.H"
    #include "createFieldRefs.H"

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        InfoErr << "Unable to find OpenCL platforms\n";
        return 0;
    }
    int plat = 0;
    const char *platEnv = std::getenv("FOAM_OCL_PLATFORM");
    if (platEnv) {
        plat = std::stoi(std::string(platEnv));
        if (plat >= platforms.size() || plat < 0) {
            InfoErr << "Invalid platfrom index: " << plat << endl;
            return 0;
        }
    }
    cl::Platform platform = platforms[plat];
    Info << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << '\n';

    cl_context_properties properties[] =
    { CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0};
    cl::Context context(CL_DEVICE_TYPE_GPU, properties);

    // get all devices associated with the context
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    if (devices.size() == 0) {
        InfoErr << "Zero devices for platform " << plat << endl;
        return 0;
    }
    cl::Device device = devices[0];

    std::string baseDir = std::getenv("WM_PROJECT_DIR");
    std::ifstream srcFile(baseDir + "/src/OpenFOAM/matrices/lduMatrix/kernels.cl");
    std::string kernelSrc((std::istreambuf_iterator<char>(srcFile)),
                    std::istreambuf_iterator<char>());

    cl::Program program(context, kernelSrc);
    cl_int err = program.build(devices);
    if (err != CL_SUCCESS) {
        InfoErr << "OpenCL program build error: " << err << endl;
        std::string msg = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        InfoErr << msg;
        return 1;
    }

    cl::CommandQueue queue(context, device);
    OpenCL opencl{platform, device, context, program, queue};
    bool useGPU = false;
    const char *gpuEnv = std::getenv("FOAM_USE_GPU");
    if (gpuEnv) {
        useGPU = std::stoi(std::string(gpuEnv));
    }

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nEvolving thermodynamics\n" << endl;
    double start = omp_get_wtime();
    if (mesh.solutionDict().found("SIMPLE"))
    {
        simpleControl simple(mesh);

        while (simple.loop())
        {
            Info<< "Time = " << runTime.timeName() << nl << endl;

            while (simple.correctNonOrthogonal())
            {
                #include "hEqn.H"
            }

            runTime.write();

            runTime.printExecutionTime(Info);
        }
    }
    else
    {
        pimpleControl pimple(mesh);

        #include "createDyMControls.H"

        while (runTime.run())
        {
            #include "readDyMControls.H"
            #include "readSolidTimeControls.H"

            #include "solidDiffusionNo.H"
            #include "setMultiRegionDeltaT.H"

            ++runTime;

            Info<< "Time = " << runTime.timeName() << nl << endl;

            while (pimple.loop())
            {
                if (pimple.firstIter() || moveMeshOuterCorrectors)
                {
                    // Do any mesh changes
                    mesh.controlledUpdate();

                    if (mesh.changing() && checkMeshCourantNo)
                    {
                        #include "meshCourantNo.H"
                    }
                }

                while (pimple.correct())
                {
                    #include "hEqn.H"
                }
            }

            runTime.write();
            runTime.printExecutionTime(Info);
        }
    }
    double end = omp_get_wtime();
    printf("total solver time: %.4lf s\n", (end - start));

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
