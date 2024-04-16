/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
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
    scalarTransportFoam

Group
    grpBasicSolvers

Description
    Passive scalar transport equation solver.

    \heading Solver details
    The equation is given by:

    \f[
        \ddt{T} + \div \left(\vec{U} T\right) - \div \left(D_T \grad T \right)
        = S_{T}
    \f]

    Where:
    \vartable
        T       | Passive scalar
        D_T     | Diffusion coefficient
        S_T     | Source
    \endvartable

    \heading Required fields
    \plaintable
        T       | Passive scalar
        U       | Velocity [m/s]
    \endplaintable

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "fvOptions.H"
#include "simpleControl.H"
#include "omp.h"
#include "gpu.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Passive scalar transport equation solver."
    );

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"

    simpleControl simple(mesh);

    #include "createFields.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nCalculating scalar transport\n" << endl;

    #include "CourantNo.H"

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
    double start = omp_get_wtime();
    while (simple.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        while (simple.correctNonOrthogonal())
        {
            fvScalarMatrix TEqn
            (
                fvm::ddt(T)
              + fvm::div(phi, T)
              - fvm::laplacian(DT, T)
             ==
                fvOptions(T)
            );

            TEqn.relax();
            fvOptions.constrain(TEqn);
            double t1 = omp_get_wtime();
            if (useGPU)
                TEqn.solveGPU(opencl);
            else
                TEqn.solve();
            double t2 = omp_get_wtime();
            printf("solver loop time: %lf ms\n", (t2 - t1) * 1000);
            fvOptions.correct(T);
        }

        runTime.write();
    }
    double end = omp_get_wtime();
    printf("total solver time: %.4lf s\n", (end - start));

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
