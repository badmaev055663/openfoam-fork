/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2016-2017 OpenFOAM Foundation
    Copyright (C) 2019-2021 OpenCFD Ltd.
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

\*---------------------------------------------------------------------------*/

#include "PBiCGStab.H"
#include "PrecisionAdaptor.H"
#include <CL/opencl.hpp>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(PBiCGStab, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<PBiCGStab>
        addPBiCGStabSymMatrixConstructorToTable_;

    lduMatrix::solver::addasymMatrixConstructorToTable<PBiCGStab>
        addPBiCGStabAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::PBiCGStab::PBiCGStab
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    )
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

const std::string src = R"(
#define BUFFSIZE 1024
kernel void calcSa(global const double *rAPtr,
                global const double *AyAPtr,
                global double *sAPtr,
                double alpha) {
    int i = get_global_id(0);
    sAPtr[i] = rAPtr[i] - alpha * AyAPtr[i];
}
)";

struct OpenCL {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

static int local_sz = 256;
static void run_kernel(OpenCL& opencl,
                    cl::Kernel &kernel,
                    double *sAPtr,
                    double *rAPtr,
                    double *AyAPtr,
                    double alpha,
                    int n)
{
    cl::Buffer rA_buf(opencl.queue, rAPtr, rAPtr + n, true);
    cl::Buffer AyA_buf(opencl.queue, AyAPtr, AyAPtr + n, true);
    cl::Buffer sA_buf(opencl.context, CL_MEM_READ_WRITE, n * sizeof(double));
    
    kernel.setArg(0, rA_buf);
    kernel.setArg(1, AyA_buf);
    kernel.setArg(2, sA_buf);
    kernel.setArg(3, alpha);

    opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n), cl::NDRange(local_sz));
    opencl.queue.finish();
    opencl.queue.enqueueReadBuffer(sA_buf, true, 0, n * sizeof(double), sAPtr);
}

Foam::solverPerformance Foam::PBiCGStab::scalarSolve
(
    solveScalarField& psi,
    const solveScalarField& source,
    const direction cmpt
) const
{
    // --- Setup class containing solver performance data
    solverPerformance solverPerf
    (
        lduMatrix::preconditioner::getName(controlDict_) + typeName,
        fieldName_
    );
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cerr << "Unable to find OpenCL platforms\n";
        return solverPerf;
    }
    cl::Platform platform = platforms[0];
    std::clog << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << '\n';
    // create context
    cl_context_properties properties[] =
    { CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0};
    cl::Context context(CL_DEVICE_TYPE_GPU, properties);
    // get all devices associated with the context
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    cl::Device device = devices[0];
    std::clog << "Device name: " << device.getInfo<CL_DEVICE_NAME>() << '\n';
    cl::Program program(context, src);
    program.build(devices);
    cl::CommandQueue queue(context, device);
    OpenCL opencl{platform, device, context, program, queue};
    cl::Kernel kernel(opencl.program, "calcSa");

    const label nCells = psi.size();

    solveScalar* __restrict__ psiPtr = psi.begin();

    solveScalarField pA(nCells);
    solveScalar* __restrict__ pAPtr = pA.begin();

    solveScalarField yA(nCells);
    solveScalar* __restrict__ yAPtr = yA.begin();

    // --- Calculate A.psi
    matrix_.Amul(yA, psi, interfaceBouCoeffs_, interfaces_, cmpt);

    // --- Calculate initial residual field
    solveScalarField rA(source - yA);
    solveScalar* __restrict__ rAPtr = rA.begin();

    matrix().setResidualField
    (
        ConstPrecisionAdaptor<scalar, solveScalar>(rA)(),
        fieldName_,
        true
    );

    // --- Calculate normalisation factor
    const solveScalar normFactor = this->normFactor(psi, source, yA, pA);

    if ((log_ >= 2) || (lduMatrix::debug >= 2))
    {
        Info<< "   Normalisation factor = " << normFactor << endl;
    }

    // --- Calculate normalised residual norm
    solverPerf.initialResidual() =
        gSumMag(rA, matrix().mesh().comm())
       /normFactor;
    solverPerf.finalResidual() = solverPerf.initialResidual();

    // --- Check convergence, solve if not converged
    if
    (
        minIter_ > 0
     || !solverPerf.checkConvergence(tolerance_, relTol_, log_)
    )
    {
        solveScalarField AyA(nCells);
        solveScalar* __restrict__ AyAPtr = AyA.begin();

        solveScalarField sA(nCells);
        solveScalar* __restrict__ sAPtr = sA.begin();

        solveScalarField zA(nCells);
        solveScalar* __restrict__ zAPtr = zA.begin();

        solveScalarField tA(nCells);
        solveScalar* __restrict__ tAPtr = tA.begin();

        // --- Store initial residual
        const solveScalarField rA0(rA);

        // --- Initial values not used
        solveScalar rA0rA = 0;
        solveScalar alpha = 0;
        solveScalar omega = 0;

        // --- Select and construct the preconditioner
        if (!preconPtr_)
        {
            preconPtr_ = lduMatrix::preconditioner::New
            (
                *this,
                controlDict_
            );
        }

        // --- Solver iteration
        do
        {
            // --- Store previous rA0rA
            const solveScalar rA0rAold = rA0rA;

            rA0rA = gSumProd(rA0, rA, matrix().mesh().comm());

            // --- Test for singularity
            if (solverPerf.checkSingularity(mag(rA0rA)))
            {
                break;
            }

            // --- Update pA
            if (solverPerf.nIterations() == 0)
            {
                memcpy(pAPtr, rAPtr, nCells * sizeof(solveScalar));
            }
            else
            {
                // --- Test for singularity
                if (solverPerf.checkSingularity(mag(omega)))
                {
                    break;
                }

                const solveScalar beta = (rA0rA/rA0rAold)*(alpha/omega);
                #pragma omp simd
                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] =
                        rAPtr[cell] + beta*(pAPtr[cell] - omega*AyAPtr[cell]);
                }
            }

            // --- Precondition pA
            preconPtr_->precondition(yA, pA, cmpt);

            // --- Calculate AyA
            matrix_.Amul(AyA, yA, interfaceBouCoeffs_, interfaces_, cmpt);

            const solveScalar rA0AyA =
                gSumProd(rA0, AyA, matrix().mesh().comm());

            alpha = rA0rA/rA0AyA;
            label gpusz = nCells - nCells % local_sz;
            // --- Calculate sA
            run_kernel(opencl, kernel, sAPtr, rAPtr, AyAPtr, alpha, gpusz);
            printf("gpusz: %d\n", gpusz);
            for (label cell=gpusz; cell<nCells; cell++)
            {
                sAPtr[cell] = rAPtr[cell] - alpha*AyAPtr[cell];
            }

            // --- Test sA for convergence
            solverPerf.finalResidual() =
                gSumMag(sA, matrix().mesh().comm())/normFactor;

            if
            (
                solverPerf.nIterations() >= minIter_
             && solverPerf.checkConvergence(tolerance_, relTol_, log_)
            )
            {
                #pragma omp simd
                for (label cell=0; cell<nCells; cell++)
                {
                    psiPtr[cell] += alpha*yAPtr[cell];
                }

                solverPerf.nIterations()++;

                return solverPerf;
            }

            // --- Precondition sA
            preconPtr_->precondition(zA, sA, cmpt);

            // --- Calculate tA
            matrix_.Amul(tA, zA, interfaceBouCoeffs_, interfaces_, cmpt);

            const solveScalar tAtA = gSumSqr(tA, matrix().mesh().comm());

            // --- Calculate omega from tA and sA
            //     (cheaper than using zA with preconditioned tA)
            omega = gSumProd(tA, sA, matrix().mesh().comm())/tAtA;

            // --- Update solution and residual
            #pragma omp simd
            for (label cell=0; cell<nCells; cell++)
            {
                psiPtr[cell] += alpha*yAPtr[cell] + omega*zAPtr[cell];
                rAPtr[cell] = sAPtr[cell] - omega*tAPtr[cell];
            }

            solverPerf.finalResidual() =
                gSumMag(rA, matrix().mesh().comm())
               /normFactor;
        } while
        (
            (
              ++solverPerf.nIterations() < maxIter_
            && !solverPerf.checkConvergence(tolerance_, relTol_, log_)
            )
         || solverPerf.nIterations() < minIter_
        );
    }

    if (preconPtr_)
    {
        preconPtr_->setFinished(solverPerf);
    }

    matrix().setResidualField
    (
        ConstPrecisionAdaptor<scalar, solveScalar>(rA)(),
        fieldName_,
        false
    );

    return solverPerf;
}


Foam::solverPerformance Foam::PBiCGStab::solve
(
    scalarField& psi_s,
    const scalarField& source,
    const direction cmpt
) const
{
    PrecisionAdaptor<solveScalar, scalar> tpsi(psi_s);
    return scalarSolve
    (
        tpsi.ref(),
        ConstPrecisionAdaptor<solveScalar, scalar>(source)(),
        cmpt
    );
}


// ************************************************************************* //
