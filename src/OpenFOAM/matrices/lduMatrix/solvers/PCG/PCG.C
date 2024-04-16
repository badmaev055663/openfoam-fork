/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019-2023 OpenCFD Ltd.
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

#include "PCG.H"
#include "PrecisionAdaptor.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(PCG, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<PCG>
        addPCGSymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::PCG::PCG
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

Foam::solverPerformance Foam::PCG::scalarSolve
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

    label nCells = psi.size();

    solveScalar* __restrict__ psiPtr = psi.begin();

    solveScalarField pA(nCells);
    solveScalar* __restrict__ pAPtr = pA.begin();

    solveScalarField wA(nCells);
    solveScalar* __restrict__ wAPtr = wA.begin();

    solveScalar wArA = solverPerf.great_;
    solveScalar wArAold = wArA;

    // --- Calculate A.psi
    matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);

    // --- Calculate initial residual field
    solveScalarField rA(source - wA);
    solveScalar* __restrict__ rAPtr = rA.begin();

    matrix().setResidualField
    (
        ConstPrecisionAdaptor<scalar, solveScalar>(rA)(),
        fieldName_,
        true
    );

    // --- Calculate normalisation factor
    solveScalar normFactor = this->normFactor(psi, source, wA, pA);

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
            // --- Store previous wArA
            wArAold = wArA;

            // --- Precondition residual
            preconPtr_->precondition(wA, rA, cmpt);

            // --- Update search directions:
            wArA = gSumProd(wA, rA, matrix().mesh().comm());

            if (solverPerf.nIterations() == 0)
            {
                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] = wAPtr[cell];
                }
            }
            else
            {
                const solveScalar beta = wArA/wArAold;

                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] = wAPtr[cell] + beta*pAPtr[cell];
                }
            }


            // --- Update preconditioned residual
            matrix_.Amul(wA, pA, interfaceBouCoeffs_, interfaces_, cmpt);

            solveScalar wApA = gSumProd(wA, pA, matrix().mesh().comm());

            // --- Test for singularity
            if (solverPerf.checkSingularity(mag(wApA)/normFactor)) break;


            // --- Update solution and residual:

            const solveScalar alpha = wArA/wApA;

            for (label cell=0; cell<nCells; cell++)
            {
                psiPtr[cell] += alpha*pAPtr[cell];
                rAPtr[cell] -= alpha*wAPtr[cell];
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

Foam::solverPerformance Foam::PCG::scalarSolveGPU
(
    solveScalarField& psi,
    const solveScalarField& source,
    OpenCL& opencl,
    const direction cmpt
) const
{
    // --- Setup class containing solver performance data
    solverPerformance solverPerf
    (
        lduMatrix::preconditioner::getName(controlDict_) + typeName,
        fieldName_
    );

    label nCells = psi.size();
    cl::Kernel copyKernel(opencl.program, "copy");
    cl::Kernel sumProdKernel(opencl.program, "sumProd");
    cl::Kernel multAddKernel(opencl.program, "multAdd");
    cl::Kernel addMultKernel(opencl.program, "addMult");
    cl::Kernel sumMagKernel(opencl.program, "sumMag");
    cl::Kernel invKernel(opencl.program, "inverse");
    cl::Kernel multKernel(opencl.program, "mult");

    solveScalar* __restrict__ psiPtr = psi.begin();

    solveScalarField pA(nCells);
    solveScalar* __restrict__ pAPtr = pA.begin();

    solveScalarField wA(nCells);
    solveScalar* __restrict__ wAPtr = wA.begin();

    solveScalar wArA = solverPerf.great_;
    solveScalar wArAold = wArA;

    // --- Calculate A.psi
    matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);

    // --- Calculate initial residual field
    solveScalarField rA(source - wA);
    solveScalar* __restrict__ rAPtr = rA.begin();

    matrix().setResidualField
    (
        ConstPrecisionAdaptor<scalar, solveScalar>(rA)(),
        fieldName_,
        true
    );

    // --- Calculate normalisation factor
    solveScalar normFactor = this->normFactor(psi, source, wA, pA);

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
        cl::Buffer wA_buf(opencl.queue, wAPtr, wAPtr + nCells, false);
        cl::Buffer pA_buf(opencl.queue, pAPtr, pAPtr + nCells, false);
        cl::Buffer psi_buf(opencl.queue, psiPtr, psiPtr + nCells, false);
        cl::Buffer rA_buf(opencl.queue, rAPtr, rAPtr + nCells, false);

        scalar* const __restrict__ diagPtr = const_cast<scalar*>(matrix_.diag().begin());

        cl::Buffer diag_buf(opencl.queue, diagPtr, diagPtr + nCells, true);

        label* const __restrict__ uPtr = const_cast<label*>(matrix_.lduAddr().upperAddr().begin());
        label* const __restrict__ lPtr = const_cast<label*>(matrix_.lduAddr().lowerAddr().begin());

        scalar* const __restrict__ upperPtr = const_cast<scalar*>(matrix_.upper().begin());
        scalar* const __restrict__ lowerPtr = const_cast<scalar*>(matrix_.lower().begin());

        const label nFaces = matrix_.upper().size();

        cl::Buffer lower_buf(opencl.queue, lowerPtr, lowerPtr + nFaces, true);
        cl::Buffer upper_buf(opencl.queue, upperPtr, upperPtr + nFaces, true);

        cl::Buffer l_buf(opencl.queue, lPtr, lPtr + nFaces, true);
        cl::Buffer u_buf(opencl.queue, uPtr, uPtr + nFaces, true);

        scalar* __restrict__ DPtr = const_cast<scalar*>(matrix_.diag().begin());

        cl::Buffer rD_buf(opencl.queue, DPtr, DPtr + nCells, false);
        bool usePrecond = true;
        if (usePrecond) { // init diagonal precondition
            invKernel.setArg(0, rD_buf);
            invKernel.setArg(1, nCells);
            opencl.queue.enqueueNDRangeKernel(invKernel, cl::NullRange,
                        cl::NDRange(nCells - nCells % locSz), cl::NDRange(locSz));
            opencl.queue.finish();
        }

        // --- Solver iteration
        do
        {
            // --- Store previous wArA
            wArAold = wArA;

            // --- Precondition residual
            if (usePrecond) { // diagonal precondition
                diagPrecondGPU(opencl, multKernel, wA_buf, rA_buf, rD_buf, nCells);
            } else { // none precondition
                copyGPU(opencl, copyKernel, wA_buf, rA_buf, nCells);
            }
            opencl.queue.finish();

            // --- Update search directions:
            wArA = sumProdGPU(opencl, sumProdKernel, wA_buf, rA_buf, nCells);

            if (solverPerf.nIterations() == 0)
            {
                copyGPU(opencl, copyKernel, pA_buf, wA_buf, nCells);
            }
            else
            {
                const solveScalar beta = wArA/wArAold;
                multAddKernel.setArg(0, pA_buf);
                multAddKernel.setArg(1, wA_buf);
                multAddKernel.setArg(2, beta);
                multAddKernel.setArg(3, nCells);
                opencl.queue.enqueueNDRangeKernel(multAddKernel, cl::NullRange,
                                            cl::NDRange(nCells - nCells % locSz), cl::NDRange(locSz));
            }
            opencl.queue.finish();

            // --- Update preconditioned residual
            matrix_.AmulGPU(opencl, wA, wA_buf, pA_buf, diag_buf, lower_buf, upper_buf, l_buf, u_buf);

            solveScalar wApA = sumProdGPU(opencl, sumProdKernel, wA_buf, pA_buf, nCells);

            // --- Test for singularity
            if (solverPerf.checkSingularity(mag(wApA)/normFactor)) break;

            // --- Update solution and residual:

            const solveScalar alpha = wArA/wApA;
            addMultGPU(opencl, addMultKernel, pA_buf, psi_buf, alpha, nCells);
            addMultGPU(opencl, addMultKernel, wA_buf, rA_buf, -alpha, nCells);
            opencl.queue.finish();

            solverPerf.finalResidual() = sumMagGPU(opencl, sumMagKernel, rA_buf, nCells) / normFactor;
        } while
        (
            (
              ++solverPerf.nIterations() < maxIter_
            && !solverPerf.checkConvergence(tolerance_, relTol_, log_)
            )
         || solverPerf.nIterations() < minIter_
        );
        opencl.queue.enqueueReadBuffer(psi_buf, false, 0, sizeof(double) * nCells, psiPtr);
        opencl.queue.enqueueReadBuffer(rA_buf, false, 0, sizeof(double) * nCells, rAPtr);
        opencl.queue.enqueueReadBuffer(wA_buf, false, 0, nCells * sizeof(double), wAPtr);
        opencl.queue.finish();
    }

    matrix().setResidualField
    (
        ConstPrecisionAdaptor<scalar, solveScalar>(rA)(),
        fieldName_,
        false
    );

    return solverPerf;
}


Foam::solverPerformance Foam::PCG::solve
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

Foam::solverPerformance Foam::PCG::solveGPU
(
    scalarField& psi_s,
    const scalarField& source,
    OpenCL& opencl,
    const direction cmpt
) const
{
    PrecisionAdaptor<solveScalar, scalar> tpsi(psi_s);
    return scalarSolveGPU
    (
        tpsi.ref(),
        ConstPrecisionAdaptor<solveScalar, scalar>(source)(),
        opencl,
        cmpt
    );
}


// ************************************************************************* //
